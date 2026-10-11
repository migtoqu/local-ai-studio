# Local AI Studio

Modelos de IA que se ejecutan **en tu propio ordenador**, gratis, sin conexión y sin claves,
usables de cuatro formas: un chat en el navegador, un servidor MCP para agentes (Claude Code,
Claude Desktop, Cursor…), una CLI y una API HTTP.

| Tipo | Motor | Modelos instalados |
|---|---|---|
| Texto | [Ollama](https://github.com/ollama/ollama) | `gemma4:e2b-it-qat` (por defecto, ~21 tok/s) · `qwen3.5:4b` (~13 tok/s) |
| Imagen | [mflux](https://github.com/filipstrand/mflux) (MLX) | Z-Image Turbo 4-bit (por defecto, ~4–5 min a 768×512) · FLUX.2 Klein 4B (experimental) |

Velocidades medidas en un MacBook M1 con 8 GB.

---

## Inicio rápido

```bash
cd ~/Desktop/Repositorios/local-ai-studio
./start.sh
```

Se abre http://127.0.0.1:8777. Arriba eliges **Texto** o **Imagen** y el modelo.

- **Texto**: escribe y pulsa Enter. «Razonar» activa el razonamiento extendido (más lento; si el modelo no lo soporta, se ignora).
- **Imagen**: describe la imagen (mejor en inglés) o escribe tu idea en español y pulsa
  **Mejorar prompt** para que el LLM la reescriba. Elige formato y, si quieres repetir un
  resultado, una semilla. **Galería** muestra todo lo generado (se guarda en `outputs/`).

Para pararlo: `Ctrl+C` en la terminal.

---

## Usarlo desde Claude Code (MCP)

### Ya está registrado

El servidor está dado de alta a nivel de usuario en `~/.claude.json` con el nombre `local-ai`, así
que está disponible en **todos los proyectos**. Tras registrarlo hay que **abrir una sesión nueva
de Claude Code** para que cargue las herramientas.

Para comprobarlo, en una sesión nueva pide algo como:

> «Usa local-ai para listar los modelos locales»

### Herramientas disponibles

| Herramienta | Qué hace | Parámetros principales |
|---|---|---|
| `local_generate_text` | Respuesta de un LLM local | `prompt`, `system`, `model`, `think`, `temperature`, `max_tokens` |
| `local_generate_image` | Genera un PNG y lo devuelve (ruta + imagen) | `prompt`, `width`, `height`, `seed`, `steps`, `model` |
| `local_list_models` | Lista modelos de texto e imagen | — |

Ejemplos de peticiones a Claude:

> «Con local_generate_text y el modelo qwen3.5:4b, escribe 3 eslóganes para una cuadra de PRE»
>
> «Genera con local-ai una imagen horizontal de una yegua torda en un picadero al atardecer y
> guárdala en `public/img/` de este proyecto»

No hace falta tener nada arrancado: si Ollama está apagado, el MCP lo arranca solo.

### Registrarlo en otro ordenador o cliente

Con la CLI de Claude Code:

```bash
claude mcp add -s user local-ai -- ~/Desktop/Repositorios/local-ai-studio/bin/localai-mcp
```

O añade esto en la configuración MCP de tu cliente (`~/.claude.json` → `mcpServers`, o
`claude_desktop_config.json` en Claude Desktop):

```json
"local-ai": {
  "type": "stdio",
  "command": "/Users/migueltoquerobaron/Desktop/Repositorios/local-ai-studio/bin/localai-mcp",
  "args": []
}
```

Para darlo de baja: `claude mcp remove -s user local-ai`, o borra esa entrada.

---

## CLI (scripts y agentes sin MCP)

```bash
bin/localai text "Escribe un eslogan para una cuadra de PRE"
bin/localai text "Resume en 3 viñetas" --model qwen3.5:4b --system "Eres un copywriter ecuestre"
cat ficha.txt | bin/localai text - --system "Corrige la ortografía"
bin/localai image "grey andalusian mare trotting, golden hour" --width 768 --height 512 --seed 7
bin/localai models
```

`text` imprime solo la respuesta e `image` solo la ruta del PNG, así que se pueden encadenar:

```bash
cp "$(bin/localai image 'black friesian stallion portrait')" ~/Desktop/
```

## API HTTP (con `./start.sh` en marcha)

```bash
curl -s localhost:8777/api/text -H 'content-type: application/json' \
  -d '{"prompt":"Hola","model":"qwen3.5:4b"}'
curl -sN localhost:8777/api/image -H 'content-type: application/json' \
  -d '{"prompt":"a foal in a meadow","width":640,"height":640}'   # NDJSON con el progreso
```

Otros endpoints: `GET /api/status`, `GET /api/images`, `POST /api/chat` (streaming NDJSON).

## Python

```python
import sys; sys.path.insert(0, "/Users/migueltoquerobaron/Desktop/Repositorios/local-ai-studio")
from localai import generate_text, generate_image

generate_text("Hola", model="qwen3.5:4b")           # -> str
generate_image("a grey horse", width=768, height=512)  # -> Path del PNG
```

---

## Añadir o cambiar modelos

- **Texto**: `ollama pull <modelo>` y aparecerá solo en el chat y en las herramientas. En 8 GB
  solo caben modelos de ~4–5 GB. Por ejemplo, `gemma4:e4b` **no cabe** (pesa 9,5 GB en Mac).
  El orden de preferencia está en `PREFERRED_TEXT_MODELS` de `localai/core.py`.
- **Imagen**: se añaden en `IMAGE_MODELS` de `localai/core.py`. Los pesos se descargan en
  `~/.cache/huggingface` la primera vez que se usan.

## Estructura

```
localai/core.py        núcleo: generate_text / generate_image (lo usa todo lo demás)
localai/server.py      servidor web + API HTTP (puerto 8777)
localai/mcp_server.py  servidor MCP (stdio)
localai/cli.py         CLI
web/index.html         chat
bin/                   lanzadores (localai, localai-web, localai-mcp)
outputs/               imágenes generadas (ignorado por git)
```

## Notas y solución de problemas

- **Memoria (8 GB)**: antes de generar una imagen se liberan de memoria los modelos de texto, y
  un bloqueo (`outputs/.gpu.lock`) impide que se generen dos imágenes a la vez desde la web, el
  MCP o la CLI. Cierra el navegador y otras aplicaciones pesadas para ir más rápido.
- **«model requires X GiB but only Y GiB are available»**: el modelo no cabe en memoria. Usa uno
  más pequeño o cierra aplicaciones.
- **Las herramientas MCP no aparecen**: abre una sesión nueva de Claude Code y comprueba que
  `bin/localai-mcp` es ejecutable (`chmod +x bin/*`).
- **Variables de entorno**: `LOCALAI_PORT` (puerto), `LOCALAI_OUTPUT_DIR` (carpeta de imágenes),
  `OLLAMA_URL` (otra instancia de Ollama).
- **Instalación desde cero**: `brew install ollama uv`, `uv tool install mflux`,
  `ollama pull gemma4:e2b-it-qat`, `ollama pull qwen3.5:4b`, y `uv sync` en esta carpeta.
- Se ejecuta con `python -m` desde `bin/`, sin instalación editable: en carpetas sincronizadas
  con iCloud, macOS marca los `.pth` como ocultos y Python 3.13 los ignora. La dependencia `mcp`
  está fijada a `<2` porque la versión 2 cambió la API.
