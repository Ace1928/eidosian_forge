from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_special_denom():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert special_denom(Poly(1, t), Poly(t ** 2, t), Poly(1, t), Poly(t ** 2 - 1, t), Poly(t, t), DE) == (Poly(1, t), Poly(t ** 2 - 1, t), Poly(t ** 2 - 1, t), Poly(t, t))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-2 * x * t0, t0), Poly(I * k * t1, t1)]})
    DE.decrement_level()
    assert special_denom(Poly(1, t0), Poly(I * k, t0), Poly(1, t0), Poly(t0, t0), Poly(1, t0), DE) == (Poly(1, t0, domain='ZZ'), Poly(I * k, t0, domain='ZZ_I[k,x]'), Poly(t0, t0, domain='ZZ'), Poly(1, t0, domain='ZZ'))
    assert special_denom(Poly(1, t), Poly(t ** 2, t), Poly(1, t), Poly(t ** 2 - 1, t), Poly(t, t), DE, case='tan') == (Poly(1, t, t0, domain='ZZ'), Poly(t ** 2, t0, t, domain='ZZ[x]'), Poly(t, t, t0, domain='ZZ'), Poly(1, t0, domain='ZZ'))
    raises(ValueError, lambda: special_denom(Poly(1, t), Poly(t ** 2, t), Poly(1, t), Poly(t ** 2 - 1, t), Poly(t, t), DE, case='unrecognized_case'))