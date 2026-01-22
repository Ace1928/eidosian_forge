from __future__ import annotations
import bisect as bs
from typing import TYPE_CHECKING
def find_ge(a: list[float], x: float) -> int:
    """Find leftmost item greater than or equal to x."""
    i = bs.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError