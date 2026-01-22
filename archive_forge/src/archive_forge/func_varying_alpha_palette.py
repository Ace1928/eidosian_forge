from __future__ import annotations
import logging # isort:skip
import math
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from .colors.util import RGB, NamedColor
def varying_alpha_palette(color: str, n: int | None=None, start_alpha: int=0, end_alpha: int=255) -> Palette:
    """ Generate a palette that is a single color with linearly varying alpha.

    Alpha may vary from low to high or high to low, depending on the values of
    ``start_alpha`` and ``end_alpha``.

    Args:
        color (str) :
            Named color or RGB(A) hex color string. Any alpha component is
            combined with the ``start_alpha`` to ``end_alpha`` range by
            multiplying them together, so it is the maximum possible alpha that
            can be obtained.

        n (int, optional) :
            The size of the palette to generate. If not specified uses the
            maximum number of colors such that adjacent colors differ by an
            alpha of 1.

        start_alpha (int, optional) :
            The alpha component of the start of the palette is this value (in
            the range 0 to 255) multiplied by the alpha component of the
            ``color`` argument.

        end_alpha (int, optional) :
            The alpha component of the end of the palette is this value (in
            the range 0 to 255) multiplied by the alpha component of the
            ``color`` argument.

    Returns:
        seq[str] : a sequence of hex RGBA color strings

    Raises:
        ValueError if ``color`` is not recognisable as a string name or hex
            RGB(A) string, or if ``start_alpha`` or ``end_alpha`` are outside
            the range 0 to 255 inclusive.

    """
    if not 0 <= start_alpha <= 255:
        raise ValueError(f'start_alpha {start_alpha} must be in the range 0 to 255')
    if not 0 <= end_alpha <= 255:
        raise ValueError(f'end_alpha {end_alpha} must be in the range 0 to 255')
    rgba = NamedColor.from_string(color).copy()
    if rgba.a < 1.0:
        start_alpha = round(start_alpha * rgba.a)
        end_alpha = round(end_alpha * rgba.a)
    if n is None or n < 1:
        nn = int(abs(end_alpha - start_alpha)) + 1
    else:
        nn = n
    norm_start_alpha = start_alpha / 255.0
    norm_end_alpha = end_alpha / 255.0

    def set_alpha(rgba: RGB, i: int) -> RGB:
        rgba.a = norm_start_alpha + (norm_end_alpha - norm_start_alpha) * i / (nn - 1.0)
        return rgba
    palette = tuple((set_alpha(rgba, i).to_hex() for i in range(nn)))
    return palette