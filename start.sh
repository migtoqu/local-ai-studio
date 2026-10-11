#!/usr/bin/env bash
# Arranca Ollama (si no está corriendo) y el chat web en http://127.0.0.1:8777
set -e
cd "$(dirname "$0")"
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Arrancando Ollama…"
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve >/tmp/ollama.log 2>&1 &
  until curl -s http://127.0.0.1:11434/api/tags >/dev/null; do sleep 0.5; done
fi
(sleep 2 && open http://127.0.0.1:8777) &
exec ./bin/localai-web
