from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_to_dict(f, u, K=None, zero=False):
    """
    Convert a ``K[X]`` polynomial to a ``dict````.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_to_dict

    >>> dmp_to_dict([[1, 0], [], [2, 3]], 1)
    {(0, 0): 3, (0, 1): 2, (2, 1): 1}
    >>> dmp_to_dict([], 0)
    {}

    """
    if not u:
        return dup_to_dict(f, K, zero=zero)
    if dmp_zero_p(f, u) and zero:
        return {(0,) * (u + 1): K.zero}
    n, v, result = (dmp_degree(f, u), u - 1, {})
    if n == -oo:
        n = -1
    for k in range(0, n + 1):
        h = dmp_to_dict(f[n - k], v)
        for exp, coeff in h.items():
            result[(k,) + exp] = coeff
    return result