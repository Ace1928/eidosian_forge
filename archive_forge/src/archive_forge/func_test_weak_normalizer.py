from sympy.core.numbers import (I, Rational, oo)
from sympy.core.symbol import symbols
from sympy.polys.polytools import Poly
from sympy.integrals.risch import (DifferentialExtension,
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.testing.pytest import raises
from sympy.abc import x, t, z, n
def test_weak_normalizer():
    a = Poly((1 + x) * t ** 5 + 4 * t ** 4 + (-1 - 3 * x) * t ** 3 - 4 * t ** 2 + (-2 + 2 * x) * t, t)
    d = Poly(t ** 4 - 3 * t ** 2 + 2, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    r = weak_normalizer(a, d, DE, z)
    assert r == (Poly(t ** 5 - t ** 4 - 4 * t ** 3 + 4 * t ** 2 + 4 * t - 4, t, domain='ZZ[x]'), (Poly((1 + x) * t ** 2 + x * t, t, domain='ZZ[x]'), Poly(t + 1, t, domain='ZZ[x]')))
    assert weak_normalizer(r[1][0], r[1][1], DE) == (Poly(1, t), r[1])
    r = weak_normalizer(Poly(1 + t ** 2), Poly(t ** 2 - 1, t), DE, z)
    assert r == (Poly(t ** 4 - 2 * t ** 2 + 1, t), (Poly(-3 * t ** 2 + 1, t), Poly(t ** 2 - 1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (Poly(1, t), r[1])
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2)]})
    r = weak_normalizer(Poly(1 + t ** 2), Poly(t, t), DE, z)
    assert r == (Poly(t, t), (Poly(0, t), Poly(1, t)))
    assert weak_normalizer(r[1][0], r[1][1], DE, z) == (Poly(1, t), r[1])