from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def diverging_palette(palette1: Palette, palette2: Palette, n: int, midpoint: float=0.5) -> Palette:
    """ Generate a new palette by combining exactly two input palettes.

    Given an input ``palette1`` and ``palette2``, take a combined ``n`` colors,
    and combine input palettes at the relative ``midpoint``.
    ``palette1`` and ``palette2`` are meant to be sequential palettes that proceed
    left to right from perceptually dark to light colors.  In that case the returned
    palette is comprised of the input palettes connected at perceptually light ends.
    Palettes are combined by piecewise linear interpolation.

    Args:

        palette1 (seq[str]) :
            A sequence of hex RGB color strings for the first palette

        palette2 (seq[str]) :
            A sequence of hex RGB color strings for the second palette

        n (int) :
            The size of the output palette to generate

        midpoint (float, optional) :
            Relative position in the returned palette where input palettes are
            connected (default: 0.5)

    Returns:
            seq[str] : a sequence of hex RGB color strings

    Raises:
        ValueError if n is greater than the possible combined length the input palettes
    """
    palette2 = palette2[::-1]
    n1 = int(round(midpoint * n))
    n2 = int(round((1 - midpoint) * n))
    return linear_palette(palette1, n1) + linear_palette(palette2, n2)