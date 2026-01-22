from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_terms_gcd(f, K):
    """
    Remove GCD of terms from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_terms_gcd

    >>> f = ZZ.map([1, 0, 1, 0, 0])

    >>> dup_terms_gcd(f, ZZ)
    (2, [1, 0, 1])

    """
    if dup_TC(f, K) or not f:
        return (0, f)
    i = 0
    for c in reversed(f):
        if not c:
            i += 1
        else:
            break
    return (i, f[:-i])