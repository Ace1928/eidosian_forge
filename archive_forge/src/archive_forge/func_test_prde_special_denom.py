from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_prde_special_denom():
    a = Poly(t + 1, t)
    ba = Poly(t ** 2, t)
    bd = Poly(1, t)
    G = [(Poly(t, t), Poly(1, t)), (Poly(t ** 2, t), Poly(1, t)), (Poly(t ** 3, t), Poly(1, t))]
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert prde_special_denom(a, ba, bd, G, DE) == (Poly(t + 1, t), Poly(t ** 2, t), [(Poly(t, t), Poly(1, t)), (Poly(t ** 2, t), Poly(1, t)), (Poly(t ** 3, t), Poly(1, t))], Poly(1, t))
    G = [(Poly(t, t), Poly(1, t)), (Poly(1, t), Poly(t, t))]
    assert prde_special_denom(Poly(1, t), Poly(t ** 2, t), Poly(1, t), G, DE) == (Poly(1, t), Poly(t ** 2 - 1, t), [(Poly(t ** 2, t), Poly(1, t)), (Poly(1, t), Poly(1, t))], Poly(t, t))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-2 * x * t0, t0)]})
    DE.decrement_level()
    G = [(Poly(t, t), Poly(t ** 2, t)), (Poly(2 * t, t), Poly(t, t))]
    assert prde_special_denom(Poly(5 * x * t + 1, t), Poly(t ** 2 + 2 * x ** 3 * t, t), Poly(t ** 3 + 2, t), G, DE) == (Poly(5 * x * t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t, t), Poly(t ** 2, t)), (Poly(2 * t, t), Poly(t, t))], Poly(1, x))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly((t ** 2 + 1) * 2 * x, t)]})
    G = [(Poly(t + x, t), Poly(t * x, t)), (Poly(2 * t, t), Poly(x ** 2, x))]
    assert prde_special_denom(Poly(5 * x * t + 1, t), Poly(t ** 2 + 2 * x ** 3 * t, t), Poly(t ** 3, t), G, DE) == (Poly(5 * x * t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t + x, t), Poly(x * t, t)), (Poly(2 * t, t, x), Poly(x ** 2, t, x))], Poly(1, t))
    assert prde_special_denom(Poly(t + 1, t), Poly(t ** 2, t), Poly(t ** 3, t), G, DE) == (Poly(t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t + x, t), Poly(x * t, t)), (Poly(2 * t, t, x), Poly(x ** 2, t, x))], Poly(1, t))