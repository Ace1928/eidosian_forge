from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_nth(f, n, u, K):
    """
    Return the ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_nth

    >>> f = ZZ.map([[1], [2], [3]])

    >>> dmp_nth(f, 0, 1, ZZ)
    [3]
    >>> dmp_nth(f, 4, 1, ZZ)
    []

    """
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    elif n >= len(f):
        return dmp_zero(u - 1)
    else:
        return f[dmp_degree(f, u) - n]