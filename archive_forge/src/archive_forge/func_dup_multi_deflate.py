from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dup_multi_deflate(polys, K):
    """
    Map ``x**m`` to ``y`` in a set of polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_multi_deflate

    >>> f = ZZ.map([1, 0, 2, 0, 3])
    >>> g = ZZ.map([4, 0, 0])

    >>> dup_multi_deflate((f, g), ZZ)
    (2, ([1, 2, 3], [4, 0]))

    """
    G = 0
    for p in polys:
        if dup_degree(p) <= 0:
            return (1, polys)
        g = 0
        for i in range(len(p)):
            if not p[-i - 1]:
                continue
            g = igcd(g, i)
            if g == 1:
                return (1, polys)
        G = igcd(G, g)
    return (G, tuple([p[::G] for p in polys]))