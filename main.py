"""Streamscribe - paste video links, stream audio through RAM, transcribe, title, and ask questions.
Bring-your-own-keys: keys arrive per request from the browser and are never stored or logged.
Run locally:  pip install -r requirements.txt  &&  uvicorn main:app --reload   (needs ffmpeg on PATH)
"""
import asyncio, json, os, random, re, subprocess, tempfile, time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

WORKERS = int(os.getenv("WORKERS", "2"))
MIN_GAP = float(os.getenv("MIN_START_GAP", "3"))
MAX_GAP = float(os.getenv("MAX_START_GAP", "7"))
RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MAX_URLS = int(os.getenv("MAX_URLS", "100"))
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT_CHARS", "80000"))

app = FastAPI()
HERE = Path(__file__).parent


class Cfg(BaseModel):
    stt_base: str
    stt_key: str
    stt_model: str
    llm_base: str
    llm_key: str
    llm_model: str
    composio_key: str = ""
    composio_user: str = ""
    composio_account: str = ""
    mem0_user: str = "streamscribe"
    mem0_mode: str = "full"  # "full" = store the whole transcript, "facts" = let Mem0 extract key facts


class MemReq(BaseModel):
    cfg: Cfg
    items: list[dict]


class RunReq(BaseModel):
    urls: list[str]
    cfg: Cfg
    cookies: str = ""


class AskReq(BaseModel):
    question: str
    context: str
    cfg: Cfg
    history: list[dict] = []


def get_audio(url, cookies_path):
    """yt-dlp -> ffmpeg, all in memory. Returns mp3 bytes (mono, 16 kHz)."""
    cmd = ["yt-dlp", "-q", "--no-warnings", "--no-progress", "-f", "ba"]
    if cookies_path:
        cmd += ["--cookies", cookies_path]
    cmd += ["-o", "-", url]
    ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-vn",
          "-ac", "1", "-ar", "16000", "-b:a", "32k", "-f", "mp3", "pipe:1"]
    dl = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cv = subprocess.Popen(ff, stdin=dl.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    dl.stdout.close()
    try:
        out, _ = cv.communicate(timeout=180)
        err = dl.stderr.read().decode(errors="ignore").strip()
        dl.wait(timeout=10)
    except subprocess.TimeoutExpired:
        cv.kill(); dl.kill()
        raise RuntimeError("stream timed out")
    if len(out) < 5000:
        raise RuntimeError(err.splitlines()[-1] if err else "no audio received")
    return out


def transcribe(audio, c):
    r = httpx.post(f"{c.stt_base.rstrip('/')}/audio/transcriptions",
                   headers={"Authorization": f"Bearer {c.stt_key}"},
                   files={"file": ("audio.mp3", audio, "audio/mpeg")},
                   data={"model": c.stt_model}, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"speech-to-text {r.status_code}: {r.text[:200]}")
    return r.json().get("text", "").strip()


def chat(c, messages):
    r = httpx.post(f"{c.llm_base.rstrip('/')}/chat/completions",
                   headers={"Authorization": f"Bearer {c.llm_key}"},
                   json={"model": c.llm_model, "messages": messages}, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"language model {r.status_code}: {r.text[:200]}")
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


COMPOSIO_BASE = os.getenv("COMPOSIO_API_BASE", "https://backend.composio.dev")


MCP_URL = os.getenv("COMPOSIO_MCP_URL", "https://connect.composio.dev/mcp")
_mcp_sessions = {}


def _mcp_post(key, payload, sid=None):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
         "x-consumer-api-key": key}
    if sid:
        h["mcp-session-id"] = sid
    return httpx.post(MCP_URL, headers=h, json=payload, timeout=90)


def _mcp_json(r):
    if r.status_code >= 400:
        raise RuntimeError(f"Composio MCP {r.status_code}: {r.text[:160]}")
    if not r.content.strip():
        return {}
    if "text/event-stream" in r.headers.get("content-type", ""):
        for line in r.text.splitlines():
            if line.startswith("data:"):
                try:
                    return json.loads(line[5:].strip())
                except ValueError:
                    pass
        return {}
    return r.json()


def _mcp_session(key):
    if key in _mcp_sessions:
        return _mcp_sessions[key]
    r = _mcp_post(key, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "streamscribe", "version": "1.0"}}})
    _mcp_json(r)
    sid = r.headers.get("mcp-session-id", "")
    _mcp_post(key, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    _mcp_sessions[key] = sid
    return sid


def mcp_add_memory(c, args):
    """Run MEM0_ADD_NEW_MEMORY_RECORDS through the Composio MCP server (works with ck_ keys)."""
    item = {"tool_slug": "MEM0_ADD_NEW_MEMORY_RECORDS", "arguments": args}
    if c.composio_account:
        item["account"] = c.composio_account
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "COMPOSIO_MULTI_EXECUTE_TOOL",
        "arguments": {"sync_response_to_workbench": False, "thought": "Save a video transcript to memory", "tools": [item]}}}
    for attempt in (0, 1):
        r = _mcp_post(c.composio_key, call, _mcp_session(c.composio_key))
        if r.status_code in (400, 404) and attempt == 0:  # stale session: start a new one
            _mcp_sessions.pop(c.composio_key, None)
            continue
        break
    msg = _mcp_json(r)
    if msg.get("error"):
        raise RuntimeError(str(msg["error"].get("message", msg["error"]))[:200])
    res = msg.get("result") or {}
    text = "".join(p.get("text", "") for p in res.get("content", []) if isinstance(p, dict))
    if res.get("isError"):
        raise RuntimeError(text[:200] or "Composio tool error")
    try:
        data = json.loads(text)
    except ValueError:
        return
    inner = (data.get("data") or {}).get("results") or []
    first = inner[0].get("response", {}) if inner else {}
    if data.get("successful") is False or first.get("successful") is False:
        raise RuntimeError("Mem0: " + str(first.get("error") or data.get("error") or text)[:200])


def save_memory(c, title, url, text):
    """Save one transcript to Mem0 through Composio (tool MEM0_ADD_NEW_MEMORY_RECORDS)."""
    if c.composio_key.startswith("ck_"):  # MCP key: use the Composio MCP server
        mcp_add_memory(c, {"messages": [{"role": "user", "content": f"Video: {title}\nSource: {url}\n\n{text}"}],
                           "user_id": c.mem0_user or "streamscribe", "infer": c.mem0_mode == "facts"})
        return
    body = {"version": "latest", "arguments": {
        "messages": [{"role": "user", "content": f"Video: {title}\nSource: {url}\n\n{text}"}],
        "user_id": c.mem0_user or "streamscribe",
        "infer": c.mem0_mode == "facts",
    }}
    if c.composio_account:
        body["connected_account_id"] = c.composio_account
    if c.composio_user:
        body["user_id"] = c.composio_user
    r = httpx.post(f"{COMPOSIO_BASE}/api/v3/tools/execute/MEM0_ADD_NEW_MEMORY_RECORDS",
                   headers={"x-api-key": c.composio_key}, json=body, timeout=60)
    j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code == 401:
        raise RuntimeError("Composio rejected this key. Check it in dashboard.composio.dev.")
    if r.status_code >= 400 or j.get("successful") is False:
        err = j.get("error") or j.get("message") or r.text[:200]
        raise RuntimeError(f"Composio {r.status_code}: {str(err)[:200]}")


def parse_polished(out, raw):
    """Split the model reply into (title, text). Never returns an empty or junk title like '.'."""
    m = re.match(r"\s*TITLE:\s*(.+?)\s*\n+(.*)", out, re.S | re.I)
    title, text = (m.group(1), m.group(2)) if m and m.group(2).strip() else ("", out)
    text = (text or raw).strip()
    title = re.sub(r"[*_`#\"“”]+", "", title).strip(" .-:")
    if len(re.sub(r"\W", "", title)) < 3:
        title = " ".join(text.split()[:6]).strip(" .-:")
    if len(re.sub(r"\W", "", title)) < 3:
        title = "Untitled video"
    return title[:80], text


POLISH = ("You are a careful transcript editor. Fix spelling, grammar, punctuation and mis-heard words in this "
          "speech-to-text output and split it into clean paragraphs. Keep the original language and meaning; do not "
          "summarize, translate or add anything. Use plain text only, no markdown symbols. Reply in exactly this format:\n"
          "TITLE: <3 to 7 words describing the video>\n\n<cleaned transcript>")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.post("/api/run")
async def run(req: RunReq):
    urls = list(dict.fromkeys(u.strip() for u in req.urls if u.strip().startswith("http")))[:MAX_URLS]
    q, sem, gate, nxt = asyncio.Queue(), asyncio.Semaphore(WORKERS), asyncio.Lock(), [0.0]
    ck = None
    if req.cookies.strip():
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write(req.cookies); f.close(); ck = f.name

    async def pace():  # global start gap with jitter, shared by all workers
        async with gate:
            wait = nxt[0] - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            nxt[0] = time.time() + random.uniform(MIN_GAP, MAX_GAP)

    async def work(i, url):
        def ev(**k):
            q.put_nowait({"i": i, **k})
        async with sem:
            try:
                audio = None
                for a in range(RETRIES):
                    await pace(); ev(status="streaming")
                    try:
                        audio = await asyncio.to_thread(get_audio, url, ck); break
                    except Exception:
                        if a == RETRIES - 1:
                            raise
                        await asyncio.sleep(8 * 2 ** a + random.uniform(0, 4))
                ev(status="transcribing")
                raw = await asyncio.to_thread(transcribe, audio, req.cfg)
                if not raw:
                    ev(status="done", title="No speech detected", text=""); return
                ev(status="polishing")
                out = await asyncio.to_thread(chat, req.cfg, [{"role": "system", "content": POLISH},
                                                              {"role": "user", "content": raw}])
                title, text = parse_polished(out, raw)
                ev(status="done", title=title, text=text)
                if req.cfg.composio_key:
                    try:
                        await asyncio.to_thread(save_memory, req.cfg, title, url, text)
                        ev(mem="saved")
                    except Exception as e:
                        ev(mem="failed: " + str(e)[:160])
            except Exception as e:
                ev(status="error", error=str(e)[:300])

    async def gen():
        tasks = [asyncio.create_task(work(i, u)) for i, u in enumerate(urls)]
        try:
            while not (q.empty() and all(t.done() for t in tasks)):
                try:
                    yield json.dumps(await asyncio.wait_for(q.get(), 1)) + "\n"
                except asyncio.TimeoutError:
                    pass
        finally:
            for t in tasks:
                t.cancel()
            if ck:
                os.unlink(ck)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/memory")
async def memory(r: MemReq):
    if not r.cfg.composio_key:
        raise HTTPException(400, "Add your Composio API key first.")
    out = []
    for it in r.items[:200]:
        try:
            await asyncio.to_thread(save_memory, r.cfg, it.get("title", ""), it.get("url", ""), it.get("text", ""))
            out.append("saved")
        except Exception as e:
            out.append("failed: " + str(e)[:160])
    return {"results": out}


@app.post("/api/ask")
async def ask(r: AskReq):
    sys = ("Answer using only the files provided (video transcripts and documents). Refer to files by their titles. "
           "If the transcripts don't cover the question, say so. Write clear, plain prose in short paragraphs, like a written reply to a colleague. "
           "Never use markdown: no asterisks, hashes, dashes or numbers as bullets, horizontal rules, or tables.")
    try:
        hist = [{"role": m["role"], "content": m["content"][:4000]} for m in r.history[-8:]
                if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
        ans = await asyncio.to_thread(chat, r.cfg, [
            {"role": "system", "content": sys}, *hist,
            {"role": "user", "content": f"FILES:\n{r.context[:MAX_CONTEXT]}\n\nQUESTION: {r.question}"}])
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"answer": ans}
