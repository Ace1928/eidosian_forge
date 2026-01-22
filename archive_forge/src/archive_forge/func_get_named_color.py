from __future__ import annotations
import typing
from ._colormaps import ColorMap
from ._named_color_values import CRAYON, CSS4, SHORT, XKCD
def get_named_color(name: str) -> RGBHexColor:
    """
    Return the Hex code of a color
    """
    if name.startswith('#'):
        return name
    else:
        return NAMED_COLORS[name.lower()]