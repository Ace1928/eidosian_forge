from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_strip(f, u):
    """
    Remove leading zeros from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_strip

    >>> dmp_strip([[], [0, 1, 2], [1]], 1)
    [[0, 1, 2], [1]]

    """
    if not u:
        return dup_strip(f)
    if dmp_zero_p(f, u):
        return f
    i, v = (0, u - 1)
    for c in f:
        if not dmp_zero_p(c, v):
            break
        else:
            i += 1
    if i == len(f):
        return dmp_zero(u)
    else:
        return f[i:]