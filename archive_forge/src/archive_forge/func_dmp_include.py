from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_include(f, J, u, K):
    """
    Include useless levels in ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_include

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_include(f, [2], 1, ZZ)
    [[[1]], [[1], [2]]]

    """
    if not J:
        return f
    F, f = (dmp_to_dict(f, u), {})
    for monom, coeff in F.items():
        monom = list(monom)
        for j in J:
            monom.insert(j, 0)
        f[tuple(monom)] = coeff
    u += len(J)
    return dmp_from_dict(f, u, K)