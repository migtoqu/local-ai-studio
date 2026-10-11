"""Servidor web local: sirve el chat y expone una API HTTP sencilla.

    bin/localai-web   -> http://127.0.0.1:8777
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Iterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import core

WEB_DIR = core.ROOT / "web"
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

app = FastAPI(title="Local AI Studio")


def ndjson(events: Iterator[dict]) -> StreamingResponse:
    def gen():
        try:
            for e in events:
                yield json.dumps(e, ensure_ascii=False) + "\n"
        except core.LocalAIError as e:
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------------------------------------------------------------- estado


def _weights_cached(repo: str) -> bool:
    if "/" not in repo:
        return False  # modelo integrado de mflux: se descarga en la primera ejecución
    return (HF_CACHE / f"models--{repo.replace('/', '--')}").exists()


@app.get("/api/status")
def status():
    try:
        text_models = core.list_text_models()
        ollama_error = None
    except core.LocalAIError as e:
        text_models, ollama_error = [], str(e)
    image_models = []
    for spec in core.IMAGE_MODELS.values():
        try:
            core._find_binary(spec.command)
            installed = True
        except core.LocalAIError:
            installed = False
        image_models.append({
            "key": spec.key,
            "label": spec.label,
            "installed": installed,
            "weights_cached": _weights_cached(spec.model),
            "default": spec.key == core.DEFAULT_IMAGE_MODEL,
        })
    return {
        "ollama": {"ok": ollama_error is None, "error": ollama_error, "models": text_models},
        "image_models": image_models,
        "preferred_text_models": core.PREFERRED_TEXT_MODELS,
    }


# ---------------------------------------------------------------- texto


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str | None = None
    think: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    return ndjson(core.chat_stream(req.messages, req.model, req.think, req.temperature, req.max_tokens))


class TextRequest(BaseModel):
    prompt: str
    model: str | None = None
    system: str | None = None
    think: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/api/text")
def text(req: TextRequest):
    """Endpoint no-streaming pensado para agentes / scripts."""
    try:
        return {"text": core.generate_text(**req.model_dump())}
    except core.LocalAIError as e:
        raise HTTPException(502, str(e))


# ---------------------------------------------------------------- imagen


class ImageRequest(BaseModel):
    prompt: str
    model: str = core.DEFAULT_IMAGE_MODEL
    width: int = 768
    height: int = 512
    steps: int | None = None
    seed: int | None = None


def _image_events(req: ImageRequest) -> Iterator[dict]:
    cmd, out, spec = core.build_image_command(**req.model_dump())
    yield {"type": "queued", "model": spec.label}
    with core.gpu_lock():
        core.unload_text_models()
        yield {"type": "started"}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            log = yield from _follow_mflux(proc)
        finally:
            if proc.poll() is None:  # el cliente cerró la pestaña: no dejar a mflux huérfano
                proc.terminate()
                proc.wait()
    if proc.returncode != 0 or not out.exists():
        yield {"type": "error", "message": "mflux falló:\n" + "\n".join(log[-15:])}
        return
    yield {"type": "done", "url": f"/outputs/{out.name}", "path": str(out)}


def _follow_mflux(proc: subprocess.Popen):
    """Traduce la salida de tqdm de mflux a eventos de progreso; devuelve el log."""
    log, buf, last = [], "", None
    assert proc.stdout
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        if ch in "\r\n":
            if buf.strip():
                log.append(buf)
                m = core.PROGRESS_RE.search(buf)
                if m and "%" in buf:
                    step = (int(m.group(1)), int(m.group(2)))
                    if step != last:
                        last = step
                        yield {"type": "progress", "step": step[0], "total": step[1]}
                elif "download" in buf.lower() or "fetching" in buf.lower():
                    yield {"type": "log", "text": buf.strip()[-160:]}
            buf = ""
        else:
            buf += ch
    proc.wait()
    return log


@app.post("/api/image")
def image(req: ImageRequest):
    return ndjson(_image_events(req))


@app.get("/api/images")
def images():
    return [{"url": f"/outputs/{p.name}", "name": p.name} for p in core.list_images()]


# ---------------------------------------------------------------- estáticos

core.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=core.OUTPUT_DIR), name="outputs")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


def main():
    port = int(os.environ.get("LOCALAI_PORT", "8777"))
    print(f"Local AI Studio en http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
