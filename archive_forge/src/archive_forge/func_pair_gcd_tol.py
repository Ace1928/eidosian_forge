from __future__ import annotations
import math
from typing import Sequence
def pair_gcd_tol(a: float, b: float) -> float:
    """Calculate the Greatest Common Divisor of a and b.

        Unless b==0, the result will have the same sign as b (so that when
        b is divided by it, the result comes out positive).
        """
    while b > tol:
        a, b = (b, a % b)
    return a