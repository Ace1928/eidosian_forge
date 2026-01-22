from __future__ import annotations
from math import sqrt
import numpy as np
Computes the equivalent local invariants from the
    Weyl coordinates.

    Args:
        weyl (ndarray): Weyl coordinates.

    Returns:
        ndarray: Local equivalent coordinates [g0, g1, g3].

    Notes:
        This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003),
        but we multiply weyl coordinates by 2 since we are
        working in the reduced chamber.
    