from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def magma(n: int) -> Palette:
    """ Generate a palette of colors from the Magma palette.

    The full Magma palette that serves as input for deriving new palettes
    has 256 colors, and looks like:

    :bokeh-palette:`magma(256)`

    Args:
        n (int) : size of the palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the base palette length of 256

    Examples:

    .. code-block:: python

        >>> magma(6)
        ('#000003', '#3B0F6F', '#8C2980', '#DD4968', '#FD9F6C', '#FBFCBF')

    The resulting palette looks like: :bokeh-palette:`magma(6)`

    """
    return linear_palette(Magma256, n)