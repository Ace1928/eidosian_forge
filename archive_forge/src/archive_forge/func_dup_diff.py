from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (
from sympy.utilities import variations
from math import ceil as _ceil, log as _log
def dup_diff(f, m, K):
    """
    ``m``-th order derivative of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 1)
    3*x**2 + 4*x + 3
    >>> R.dup_diff(x**3 + 2*x**2 + 3*x + 4, 2)
    6*x + 4

    """
    if m <= 0:
        return f
    n = dup_degree(f)
    if n < m:
        return []
    deriv = []
    if m == 1:
        for coeff in f[:-m]:
            deriv.append(K(n) * coeff)
            n -= 1
    else:
        for coeff in f[:-m]:
            k = n
            for i in range(n - 1, n - m, -1):
                k *= i
            deriv.append(K(k) * coeff)
            n -= 1
    return dup_strip(deriv)