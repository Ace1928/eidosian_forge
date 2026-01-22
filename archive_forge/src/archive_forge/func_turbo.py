from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def turbo(n: int) -> Palette:
    """ Generate a palette of colors from the Turbo palette.

    Turbo is described here:

    https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html

    The full Turbo palette that serves as input for deriving new palettes
    has 256 colors, and looks like:

    :bokeh-palette:`turbo(256)`

    Args:
        n (int) : size of the palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the base palette length of 256

    Examples:

    .. code-block:: python

        >>> turbo(6)
        ('#00204C', '#31446B', '#666870', '#958F78', '#CAB969', '#FFE945')

    The resulting palette looks like: :bokeh-palette:`turbo(6)`

    """
    return linear_palette(Turbo256, n)