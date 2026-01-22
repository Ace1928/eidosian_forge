from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def inferno(n: int) -> Palette:
    """ Generate a palette of colors from the Inferno palette.

    The full Inferno palette that serves as input for deriving new palettes
    has 256 colors, and looks like:

    :bokeh-palette:`inferno(256)`

    Args:
        n (int) : size of the palette to generate

    Returns:
        seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the base palette length of 256

    Examples:

    .. code-block:: python

        >>> inferno(6)
        ('#000003', '#410967', '#932567', '#DC5039', '#FBA40A', '#FCFEA4')

    The resulting palette looks like: :bokeh-palette:`inferno(6)`

    """
    return linear_palette(Inferno256, n)