from __future__ import annotations
import bisect as bs
from typing import TYPE_CHECKING
def find_le(a: list[float], x: float) -> int:
    """Find rightmost value less than or equal to x."""
    if (i := bs.bisect_right(a, x)):
        return i - 1
    raise ValueError