from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public
from mpmath import MPContext
def field_isomorphism_pslq(a, b):
    """Construct field isomorphism using PSLQ algorithm. """
    if not a.root.is_real or not b.root.is_real:
        raise NotImplementedError("PSLQ doesn't support complex coefficients")
    f = a.minpoly
    g = b.minpoly.replace(f.gen)
    n, m, prev = (100, b.minpoly.degree(), None)
    ctx = MPContext()
    for i in range(1, 5):
        A = a.root.evalf(n)
        B = b.root.evalf(n)
        basis = [1, B] + [B ** i for i in range(2, m)] + [-A]
        ctx.dps = n
        coeffs = ctx.pslq(basis, maxcoeff=10 ** 10, maxsteps=1000)
        if coeffs is None:
            break
        if coeffs != prev:
            prev = coeffs
        else:
            break
        coeffs = [S(c) / coeffs[-1] for c in coeffs[:-1]]
        while not coeffs[-1]:
            coeffs.pop()
        coeffs = list(reversed(coeffs))
        h = Poly(coeffs, f.gen, domain='QQ')
        if f.compose(h).rem(g).is_zero:
            return coeffs
        else:
            n *= 2
    return None