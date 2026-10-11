"""Núcleo compartido: texto vía Ollama e imagen vía mflux.

Lo usan el servidor web, el servidor MCP y la CLI, así que cualquier agente
acaba llamando a las mismas dos funciones: `generate_text` y `generate_image`.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("LOCALAI_OUTPUT_DIR", ROOT / "outputs"))
LOCK_FILE = OUTPUT_DIR / ".gpu.lock"

# Orden de preferencia cuando no se indica modelo de texto. En 8 GB, gemma4:e4b
# (9,5 GB en MLX) no cabe; e2b-it-qat (4,3 GB) sí.
PREFERRED_TEXT_MODELS = ["gemma4:e2b-it-qat", "qwen3.5:4b"]


@dataclass(frozen=True)
class ImageModel:
    key: str
    label: str
    command: str
    model: str
    steps: int
    extra_args: tuple[str, ...] = field(default_factory=tuple)


IMAGE_MODELS: dict[str, ImageModel] = {
    "z-image-turbo": ImageModel(
        key="z-image-turbo",
        label="Z-Image Turbo (4-bit)",
        command="mflux-generate-z-image-turbo",
        model="mflux-community/z-image-turbo-mflux-q4",
        steps=9,
    ),
    "flux2-klein-4b": ImageModel(
        key="flux2-klein-4b",
        label="FLUX.2 Klein 4B (4-bit, experimental en 8 GB)",
        command="mflux-generate-flux2",
        model="flux2-klein-4b",
        steps=4,
        extra_args=("--quantize", "4"),
    ),
}
DEFAULT_IMAGE_MODEL = "z-image-turbo"


class LocalAIError(RuntimeError):
    pass


# ---------------------------------------------------------------- texto


def _start_ollama() -> bool:
    """Arranca `ollama serve` en segundo plano si está instalado; True si responde."""
    binary = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
    is_local = "127.0.0.1" in OLLAMA_URL or "localhost" in OLLAMA_URL
    if not is_local or not Path(binary).exists():
        return False
    env = {**os.environ, "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
    subprocess.Popen([binary, "serve"], env=env, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(40):
        time.sleep(0.5)
        try:
            httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return True
        except httpx.HTTPError:
            pass
    return False


def list_text_models() -> list[str]:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    except httpx.HTTPError as e:
        if not _start_ollama():
            raise LocalAIError(
                f"No se puede conectar con Ollama en {OLLAMA_URL}. Arráncalo con `ollama serve`."
            ) from e
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def resolve_text_model(model: str | None) -> str:
    available = list_text_models()
    if not available:
        raise LocalAIError("Ollama no tiene modelos. Descarga uno con `ollama pull qwen3.5:4b`.")
    if model:
        if model in available:
            return model
        raise LocalAIError(f"Modelo '{model}' no descargado. Disponibles: {', '.join(available)}")
    for m in PREFERRED_TEXT_MODELS:
        if m in available:
            return m
    return available[0]


def _chat_payload(messages, model, think, temperature, max_tokens, stream) -> dict:
    options: dict = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {"model": model, "messages": messages, "stream": stream, "options": options}
    if think is not None:
        payload["think"] = think
    return payload


def _rejects_think(status: int, body: str, payload: dict) -> bool:
    """Algunos modelos (p. ej. Gemma) no aceptan el parámetro `think`."""
    return status == 400 and "think" in body and "think" in payload


def _without_think(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "think"}


def chat_stream(
    messages: list[dict],
    model: str | None = None,
    think: bool | None = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[dict]:
    """Genera eventos {'type': 'thinking'|'content'|'done'|'error', ...}."""
    model = resolve_text_model(model)
    payload = _chat_payload(messages, model, think, temperature, max_tokens, True)
    with httpx.Client() as client:
        for attempt in range(2):
            with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload, timeout=None) as r:
                if r.status_code >= 400:
                    body = r.read().decode()
                    if attempt == 0 and _rejects_think(r.status_code, body, payload):
                        payload = _without_think(payload)
                        continue
                    yield {"type": "error", "message": f"Ollama {r.status_code}: {body}"}
                    return
                yield {"type": "model", "model": model}
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        yield {"type": "error", "message": chunk["error"]}
                        return
                    msg = chunk.get("message", {})
                    if msg.get("thinking"):
                        yield {"type": "thinking", "text": msg["thinking"]}
                    if msg.get("content"):
                        yield {"type": "content", "text": msg["content"]}
                    if chunk.get("done"):
                        tokens = chunk.get("eval_count", 0)
                        secs = chunk.get("eval_duration", 0) / 1e9
                        yield {
                            "type": "done",
                            "tokens": tokens,
                            "tokens_per_second": round(tokens / secs, 1) if secs else None,
                        }
                return


def generate_text(
    prompt: str,
    model: str | None = None,
    system: str | None = None,
    think: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Devuelve la respuesta completa de un modelo local de Ollama."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    model = resolve_text_model(model)
    payload = _chat_payload(messages, model, think, temperature, max_tokens, False)
    url = f"{OLLAMA_URL}/api/chat"
    r = httpx.post(url, json=payload, timeout=None)
    if _rejects_think(r.status_code, r.text, payload):
        r = httpx.post(url, json=_without_think(payload), timeout=None)
    if r.status_code >= 400:
        raise LocalAIError(f"Ollama devolvió {r.status_code}: {r.text}")
    return r.json()["message"]["content"].strip()


def unload_text_models() -> None:
    """Saca de memoria los modelos de Ollama para dejar sitio a la generación de imagen."""
    try:
        running = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=5).json().get("models", [])
        for m in running:
            httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": m["name"], "keep_alive": 0},
                timeout=30,
            )
    except httpx.HTTPError:
        pass  # Ollama apagado: nada que liberar.


# ---------------------------------------------------------------- imagen


def _find_binary(name: str) -> str:
    path = shutil.which(name) or str(Path.home() / ".local/bin" / name)
    if not Path(path).exists():
        raise LocalAIError(f"No encuentro `{name}`. Instala mflux con `uv tool install mflux`.")
    return path


@contextmanager
def gpu_lock():
    """Un solo trabajo de imagen a la vez entre procesos (web, MCP, CLI): con 8 GB no caben dos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_image_command(
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 768,
    height: int = 512,
    steps: int | None = None,
    seed: int | None = None,
) -> tuple[list[str], Path, ImageModel]:
    if model not in IMAGE_MODELS:
        raise LocalAIError(f"Modelo de imagen desconocido '{model}'. Opciones: {', '.join(IMAGE_MODELS)}")
    spec = IMAGE_MODELS[model]
    width, height = (max(256, min(1536, v)) // 16 * 16 for v in (width, height))
    seed = seed if seed is not None else int(time.time()) % 1_000_000
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:40].strip("-") or "imagen"
    out = OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.key}_{slug}_{seed}.png"
    cmd = [
        _find_binary(spec.command),
        "--model", spec.model,
        *spec.extra_args,
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "--steps", str(steps or spec.steps),
        "--seed", str(seed),
        "--low-ram",
        "--output", str(out),
    ]
    return cmd, out, spec


PROGRESS_RE = re.compile(r"(\d+)/(\d+)")


def generate_image(
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 768,
    height: int = 512,
    steps: int | None = None,
    seed: int | None = None,
    timeout: int = 1800,
) -> Path:
    """Genera una imagen en local y devuelve la ruta del PNG."""
    cmd, out, _ = build_image_command(prompt, model, width, height, steps, seed)
    with gpu_lock():
        unload_text_models()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout)[-1500:]
        raise LocalAIError(f"mflux falló (código {proc.returncode}):\n{tail}")
    return out


def list_images(limit: int = 50) -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
