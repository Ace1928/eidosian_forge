from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_slice_in(f, m, n, j, u, K):
    """Take a continuous subsequence of terms of ``f`` in ``x_j`` in ``K[X]``. """
    if j < 0 or j > u:
        raise IndexError('-%s <= j < %s expected, got %s' % (u, u, j))
    if not u:
        return dup_slice(f, m, n, K)
    f, g = (dmp_to_dict(f, u), {})
    for monom, coeff in f.items():
        k = monom[j]
        if k < m or k >= n:
            monom = monom[:j] + (0,) + monom[j + 1:]
        if monom in g:
            g[monom] += coeff
        else:
            g[monom] = coeff
    return dmp_from_dict(g, u, K)