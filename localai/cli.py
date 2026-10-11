"""CLI para scripts y agentes sin MCP.

    bin/localai text "Resume esto..." [--model qwen3.5:4b] [--system "..."] [--think]
    bin/localai image "a grey andalusian horse..." [--width 768 --height 512 --seed 7]
    bin/localai models

`text` imprime solo la respuesta; `image` imprime solo la ruta del PNG,
así que la salida se puede encadenar directamente.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import core


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="localai", description="Modelos locales de texto e imagen")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("text", help="Genera texto con Ollama")
    t.add_argument("prompt", help="Prompt; usa '-' para leer de stdin")
    t.add_argument("--model")
    t.add_argument("--system")
    t.add_argument("--think", action="store_true")
    t.add_argument("--temperature", type=float)
    t.add_argument("--max-tokens", type=int)

    i = sub.add_parser("image", help="Genera una imagen con mflux")
    i.add_argument("prompt")
    i.add_argument("--model", default=core.DEFAULT_IMAGE_MODEL, choices=list(core.IMAGE_MODELS))
    i.add_argument("--width", type=int, default=768)
    i.add_argument("--height", type=int, default=512)
    i.add_argument("--steps", type=int)
    i.add_argument("--seed", type=int)

    sub.add_parser("models", help="Lista modelos disponibles (JSON)")

    a = p.parse_args(argv)
    try:
        if a.cmd == "text":
            prompt = sys.stdin.read() if a.prompt == "-" else a.prompt
            print(core.generate_text(prompt, a.model, a.system, a.think, a.temperature, a.max_tokens))
        elif a.cmd == "image":
            print(core.generate_image(a.prompt, a.model, a.width, a.height, a.steps, a.seed))
        else:
            print(json.dumps({
                "text": core.list_text_models(),
                "image": list(core.IMAGE_MODELS),
            }, indent=2))
    except core.LocalAIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
