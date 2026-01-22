from __future__ import annotations
import typing
def get_brewer_palette(scheme: ColorScheme | ColorSchemeShort, palette: int | str) -> ColorPalette:
    """
    Return color palette from a given scheme
    """
    if isinstance(palette, int):
        palette = number_to_name(scheme, palette)
    mod = get_palette_module(scheme)
    return getattr(mod, palette)