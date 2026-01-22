from .sage_helper import _within_sage
from functools import reduce
import operator
def xgcd(a, b):
    """
        Returns a triple ``(g,s,t)`` such that `g = s\\cdot a+t\\cdot b = \\gcd(a,b)`.

        >>> xgcd(56, 44)
        (4, 4, -5)
        >>> xgcd(5, 7)
        (1, 3, -2)
        >>> xgcd(-5, -7)
        (1, -3, 2)
        >>> xgcd(5, -7)
        (1, 3, 2)
        >>> xgcd(-5, 7)
        (1, -3, -2)
        """
    old_r, r = (a, b)
    old_s, s = (1, 0)
    old_t, t = (0, 1)
    while r != 0:
        q = old_r // r
        old_r, r = (r, old_r - q * r)
        old_s, s = (s, old_s - q * s)
        old_t, t = (t, old_t - q * t)
    if old_r > 0:
        return (old_r, old_s, old_t)
    return (-old_r, -old_s, -old_t)