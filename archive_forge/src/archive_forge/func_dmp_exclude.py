from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_exclude(f, u, K):
    """
    Exclude useless levels from ``f``.

    Return the levels excluded, the new excluded ``f``, and the new ``u``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_exclude

    >>> f = ZZ.map([[[1]], [[1], [2]]])

    >>> dmp_exclude(f, 2, ZZ)
    ([2], [[1], [1, 2]], 1)

    """
    if not u or dmp_ground_p(f, None, u):
        return ([], f, u)
    J, F = ([], dmp_to_dict(f, u))
    for j in range(0, u + 1):
        for monom in F.keys():
            if monom[j]:
                break
        else:
            J.append(j)
    if not J:
        return ([], f, u)
    f = {}
    for monom, coeff in F.items():
        monom = list(monom)
        for j in reversed(J):
            del monom[j]
        f[tuple(monom)] = coeff
    u -= len(J)
    return (J, dmp_from_dict(f, u, K), u)