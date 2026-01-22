from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_nf_buchberger_reduced(f, G, O, K):
    """
    Compute a reduced normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    In contrast to weak normal forms, reduced normal forms *are* unique, but
    their computation is more expensive.

    This is the standard Buchberger algorithm for computing reduced normal forms
    with respect to *global* monomial orders [SCA, algorithm 1.6.11].

    The ``pantom`` option is not supported, so this normal form cannot be used
    as a normal form for the "extended" groebner algorithm.
    """
    h = sdm_zero()
    g = f
    while g:
        g = sdm_nf_buchberger(g, G, O, K)
        if g:
            h = sdm_add(h, [sdm_LT(g)], O, K)
            g = g[1:]
    return h