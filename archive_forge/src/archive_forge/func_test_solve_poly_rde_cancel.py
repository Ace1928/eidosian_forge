from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_solve_poly_rde_cancel():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert cancel_exp(Poly(2 * x, t), Poly(2 * x, t), 0, DE) == Poly(1, t)
    assert cancel_exp(Poly(2 * x, t), Poly((1 + 2 * x) * t, t), 1, DE) == Poly(t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    raises(NonElementaryIntegralException, lambda: cancel_primitive(Poly(1, t), Poly(t, t), oo, DE))
    assert cancel_primitive(Poly(1, t), Poly(t + 1 / x, t), 2, DE) == Poly(t, t)
    assert cancel_primitive(Poly(4 * x, t), Poly(4 * x * t ** 2 + 2 * t / x, t), 3, DE) == Poly(t ** 2, t)