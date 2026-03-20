from __future__ import annotations

from typing import List, Optional

from eidosian_core import eidosian

from ..core import tool
from ..state import FORGE_DIR
from ._param_coercion import coerce_string_list

try:
    from glyph_forge.services.image_to_glyph import image_to_glyph
    from glyph_forge.services.text_to_banner import text_to_banner
except ImportError:
    import sys

    sys.path.append(str(FORGE_DIR / "glyph_forge/src"))
    from glyph_forge.services.image_to_glyph import image_to_glyph
    from glyph_forge.services.text_to_banner import text_to_banner


@eidosian()
@tool(name="glyph_text_to_banner", description="Convert text to a styled ASCII banner.")
def glyph_text_to_banner(
    text: str,
    style: str = "minimal",
    font: Optional[str] = None,
    width: Optional[int] = None,
    effects: Optional[List[str] | str] = None,
    color: bool = False,
) -> str:
    """
    Generate a styled ASCII banner using the Glyph Forge engine.
    Supports various fonts, styles, and special effects.
    """
    return text_to_banner(
        text=text,
        style=style,
        font=font,
        width=width,
        effects=coerce_string_list(effects),
        color=color,
    )


@eidosian()
@tool(name="glyph_image_to_ascii", description="Convert an image to high-fidelity ASCII/ANSI art.")
def glyph_image_to_ascii(
    image_path: str,
    output_path: Optional[str] = None,
    style: Optional[str] = None,
    color_mode: str = "none",
    width: int = 100,
    brightness: float = 1.0,
    contrast: float = 1.0,
    dithering: bool = False,
    invert: bool = False,
    charset: str = "general",
) -> str:
    """
    High-fidelity image-to-text conversion.

    Args:
        image_path: Local path to the source image.
        output_path: Optional path to save the result.
        style: Aesthetic style to apply.
        color_mode: 'none', 'ansi', 'ansi16', 'ansi256', 'truecolor', 'html'.
        width: Output width in characters.
        brightness: Brightness factor (0.0-2.0).
        contrast: Contrast factor (0.0-2.0).
        dithering: Enable Floyd-Steinberg dithering.
        invert: Invert character density.
        charset: Character set to use ('general', 'blocks', etc.).
    """
    return image_to_glyph(
        image_path=image_path,
        output_path=output_path,
        style=style,
        color_mode=color_mode,
        width=width,
        brightness=brightness,
        contrast=contrast,
        dithering=dithering,
        invert=invert,
        charset=charset,
    )
