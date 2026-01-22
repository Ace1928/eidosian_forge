from __future__ import annotations
import bisect as bs
from typing import TYPE_CHECKING
def find_gt(a: list[float], x: float) -> int:
    """Find leftmost value greater than x."""
    i = bs.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError