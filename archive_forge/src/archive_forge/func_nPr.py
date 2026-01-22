from __future__ import annotations
import math
def nPr(n: int, r: int) -> int:
    """
    Calculates nPr.

    Args:
        n (int): total number of items.
        r (int): items to permute

    Returns:
        nPr.
    """
    f = math.factorial
    return int(f(n) / f(n - r))