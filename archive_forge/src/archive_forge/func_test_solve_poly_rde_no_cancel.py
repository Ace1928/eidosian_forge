from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_solve_poly_rde_no_cancel():
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2, t)]})
    assert solve_poly_rde(Poly(t ** 2 + 1, t), Poly(t ** 3 + (x + 1) * t ** 2 + t + x + 2, t), oo, DE) == Poly(t + x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert solve_poly_rde(Poly(0, x), Poly(x / 2 - Rational(1, 4), x), oo, DE) == Poly(x ** 2 / 4 - x / 4, x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert solve_poly_rde(Poly(2, t), Poly(t ** 2 + 2 * t + 3, t), 1, DE) == Poly(t + 1, t, x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert no_cancel_equal(Poly(1 - t, t), Poly(t ** 3 + t ** 2 - 2 * x * t - 2 * x, t), oo, DE) == (Poly(t ** 2, t), 1, Poly((-2 - 2 * x) * t - 2 * x, t))