from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_bound_degree():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert bound_degree(Poly(1, x), Poly(-2 * x, x), Poly(1, x), DE) == 0
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert bound_degree(Poly(t, t), Poly((t - 1) * (t ** 2 + 1), t), Poly(1, t), DE) == 0