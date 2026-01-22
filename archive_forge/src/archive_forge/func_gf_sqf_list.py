from math import ceil as _ceil, sqrt as _sqrt, prod
from sympy.core.random import uniform
from sympy.external.gmpy import SYMPY_INTS
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors
def gf_sqf_list(f, p, K, all=False):
    """
    Return the square-free decomposition of a ``GF(p)[x]`` polynomial.

    Given a polynomial ``f`` in ``GF(p)[x]``, returns the leading coefficient
    of ``f`` and a square-free decomposition ``f_1**e_1 f_2**e_2 ... f_k**e_k``
    such that all ``f_i`` are monic polynomials and ``(f_i, f_j)`` for ``i != j``
    are co-prime and ``e_1 ... e_k`` are given in increasing order. All trivial
    terms (i.e. ``f_i = 1``) are not included in the output.

    Consider polynomial ``f = x**11 + 1`` over ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ

       >>> from sympy.polys.galoistools import (
       ...     gf_from_dict, gf_diff, gf_sqf_list, gf_pow,
       ... )
       ... # doctest: +NORMALIZE_WHITESPACE

       >>> f = gf_from_dict({11: ZZ(1), 0: ZZ(1)}, 11, ZZ)

    Note that ``f'(x) = 0``::

       >>> gf_diff(f, 11, ZZ)
       []

    This phenomenon does not happen in characteristic zero. However we can
    still compute square-free decomposition of ``f`` using ``gf_sqf()``::

       >>> gf_sqf_list(f, 11, ZZ)
       (1, [([1, 1], 11)])

    We obtained factorization ``f = (x + 1)**11``. This is correct because::

       >>> gf_pow([1, 1], 11, 11, ZZ) == f
       True

    References
    ==========

    .. [1] [Geddes92]_

    """
    n, sqf, factors, r = (1, False, [], int(p))
    lc, f = gf_monic(f, p, K)
    if gf_degree(f) < 1:
        return (lc, [])
    while True:
        F = gf_diff(f, p, K)
        if F != []:
            g = gf_gcd(f, F, p, K)
            h = gf_quo(f, g, p, K)
            i = 1
            while h != [K.one]:
                G = gf_gcd(g, h, p, K)
                H = gf_quo(h, G, p, K)
                if gf_degree(H) > 0:
                    factors.append((H, i * n))
                g, h, i = (gf_quo(g, G, p, K), G, i + 1)
            if g == [K.one]:
                sqf = True
            else:
                f = g
        if not sqf:
            d = gf_degree(f) // r
            for i in range(0, d + 1):
                f[i] = f[i * r]
            f, n = (f[:d + 1], n * r)
        else:
            break
    if all:
        raise ValueError("'all=True' is not supported yet")
    return (lc, factors)