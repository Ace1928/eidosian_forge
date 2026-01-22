from __future__ import annotations
import typing
from ._colormaps import ColorMap
from ._named_color_values import CRAYON, CSS4, SHORT, XKCD
def get_colormap(name: str) -> ColorMap:
    """
    Return colormap
    """
    return COLORMAPS[name]