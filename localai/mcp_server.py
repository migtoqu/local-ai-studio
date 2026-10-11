"""Servidor MCP (stdio) para que cualquier agente use los modelos locales.

Registro en Claude Code:
    claude mcp add local-ai -- /ruta/a/local-ai-studio/bin/localai-mcp
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP, Image

from . import core

mcp = FastMCP(
    "local-ai",
    instructions=(
        "Modelos que corren en el propio ordenador, gratis y sin conexión. "
        "local_generate_text para texto (Ollama), local_generate_image para imágenes (mflux). "
        "La generación de imagen tarda varios minutos en un Mac de 8 GB; no la lances en paralelo."
    ),
)


@mcp.tool()
def local_generate_text(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    think: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Genera texto con un LLM local (Ollama).

    Args:
        prompt: Instrucción o pregunta para el modelo.
        system: Mensaje de sistema opcional (rol, tono, formato).
        model: Nombre del modelo de Ollama (p. ej. "qwen3.5:4b", "gemma4:e2b-it-qat").
            Si se omite se usa el mejor disponible.
        think: Activa el razonamiento extendido (más lento; solo modelos que lo soportan).
        temperature: Creatividad (0 = determinista, 1 = creativo).
        max_tokens: Límite de tokens de salida.
    """
    return core.generate_text(prompt, model, system, think, temperature, max_tokens)


@mcp.tool()
def local_generate_image(
    prompt: str,
    width: int = 768,
    height: int = 512,
    model: str = core.DEFAULT_IMAGE_MODEL,
    steps: int | None = None,
    seed: int | None = None,
    return_image: bool = True,
) -> list:
    """Genera una imagen en local con mflux y devuelve la ruta del PNG.

    Args:
        prompt: Descripción de la imagen, preferiblemente en inglés y detallada.
        width: Ancho en píxeles (múltiplo de 16, 256-1536). Más grande = más lento.
        height: Alto en píxeles (múltiplo de 16, 256-1536).
        model: "z-image-turbo" (recomendado) o "flux2-klein-4b".
        steps: Pasos de difusión; por defecto el recomendado para el modelo.
        seed: Semilla para reproducir un resultado.
        return_image: Si es True, además de la ruta devuelve la imagen para verla.
    """
    path = core.generate_image(prompt, model, width, height, steps, seed)
    result: list = [f"Imagen guardada en {path}"]
    if return_image:
        result.append(Image(path=str(path)))
    return result


@mcp.tool()
def local_list_models() -> dict:
    """Lista los modelos locales disponibles de texto (Ollama) e imagen (mflux)."""
    try:
        text = core.list_text_models()
    except core.LocalAIError as e:
        text = [f"error: {e}"]
    return {
        "text": text,
        "image": {k: v.label for k, v in core.IMAGE_MODELS.items()},
    }


def main():
    mcp.run()


if __name__ == "__main__":
    main()
