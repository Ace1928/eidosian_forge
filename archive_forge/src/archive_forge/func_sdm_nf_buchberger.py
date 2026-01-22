from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_nf_buchberger(f, G, O, K, phantom=None):
    """
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is the standard Buchberger algorithm for computing weak normal forms with
    respect to *global* monomial orders [SCA, algorithm 1.6.10].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    from itertools import repeat
    h = f
    T = list(G)
    if phantom is not None:
        hp = phantom[0]
        Tp = list(phantom[1])
        phantom = True
    else:
        Tp = repeat([])
        phantom = False
    while h:
        try:
            g, gp = next(((g, gp) for g, gp in zip(T, Tp) if sdm_monomial_divides(sdm_LM(g), sdm_LM(h))))
        except StopIteration:
            break
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            h = sdm_spoly(h, g, O, K)
    if phantom:
        return (h, hp)
    return h