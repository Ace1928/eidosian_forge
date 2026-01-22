from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_revert(f, n, K):
    """
    Compute ``f**(-1)`` mod ``x**n`` using Newton iteration.

    This function computes first ``2**n`` terms of a polynomial that
    is a result of inversion of a polynomial modulo ``x**n``. This is
    useful to efficiently compute series expansion of ``1/f``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = -QQ(1,720)*x**6 + QQ(1,24)*x**4 - QQ(1,2)*x**2 + 1

    >>> R.dup_revert(f, 8)
    61/720*x**6 + 5/24*x**4 + 1/2*x**2 + 1

    """
    g = [K.revert(dup_TC(f, K))]
    h = [K.one, K.zero, K.zero]
    N = int(_ceil(_log(n, 2)))
    for i in range(1, N + 1):
        a = dup_mul_ground(g, K(2), K)
        b = dup_mul(f, dup_sqr(g, K), K)
        g = dup_rem(dup_sub(a, b, K), h, K)
        h = dup_lshift(h, dup_degree(h), K)
    return g