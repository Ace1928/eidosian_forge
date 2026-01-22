from __future__ import annotations
import abc
from enum import Enum
from typing import TYPE_CHECKING
import numpy as np
def continuous_palette(self, x: FloatArrayLike) -> Sequence[RGBHexColor | None]:
    """
        Return colors correspondsing to proportions in x

        Parameters
        ----------
        x :
            Values in the range [0, 1]. O maps to the start of the
            gradient, and 1 to the end of the gradient.
        """
    x = np.asarray(x)
    bad_bool_idx = np.isnan(x) | np.isinf(x)
    has_bad = bad_bool_idx.any()
    if has_bad:
        x[bad_bool_idx] = 0
    hex_colors = self._generate_colors(x)
    if has_bad:
        hex_colors = [None if isbad else c for c, isbad in zip(hex_colors, bad_bool_idx)]
    return hex_colors