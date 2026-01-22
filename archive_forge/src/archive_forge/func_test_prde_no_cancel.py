from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_prde_no_cancel():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert prde_no_cancel_b_large(Poly(1, x), [Poly(x ** 2, x), Poly(1, x)], 2, DE) == ([Poly(x ** 2 - 2 * x + 2, x), Poly(1, x)], Matrix([[1, 0, -1, 0], [0, 1, 0, -1]], x))
    assert prde_no_cancel_b_large(Poly(1, x), [Poly(x ** 3, x), Poly(1, x)], 3, DE) == ([Poly(x ** 3 - 3 * x ** 2 + 6 * x - 6, x), Poly(1, x)], Matrix([[1, 0, -1, 0], [0, 1, 0, -1]], x))
    assert prde_no_cancel_b_large(Poly(x, x), [Poly(x ** 2, x), Poly(1, x)], 1, DE) == ([Poly(x, x, domain='ZZ'), Poly(0, x, domain='ZZ')], Matrix([[1, -1, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]], x))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 3 + 1, t)]})
    G = [Poly(t ** 6, t), Poly(x * t ** 5, t), Poly(t ** 3, t), Poly(x * t ** 2, t), Poly(1 + x, t)]
    R = QQ.frac_field(x)[t]
    assert prde_no_cancel_b_small(Poly(x * t, t), G, 4, DE) == ([Poly(t ** 4 / 4 - x / 12 * t ** 3 + x ** 2 / 24 * t ** 2 + (Rational(-11, 12) - x ** 3 / 24) * t + x / 24, t), Poly(x / 3 * t ** 3 - x ** 2 / 6 * t ** 2 + (Rational(-1, 3) + x ** 3 / 6) * t - x / 6, t), Poly(t, t), Poly(0, t), Poly(0, t)], Matrix([[1, 0, -1, 0, 0, 0, 0, 0, 0, 0], [0, 1, Rational(-1, 4), 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, -1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, -1]], ring=R))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t ** 2, t)]})
    b = Poly(-1 / x ** 2, t, field=True)
    q = [Poly(x ** i * t ** j, t, field=True) for i in range(2) for j in range(3)]
    h, A = prde_no_cancel_b_small(b, q, 3, DE)
    V = A.nullspace()
    R = QQ.frac_field(x)[t]
    assert len(V) == 1
    assert V[0] == Matrix([Rational(-1, 2), 0, 0, 1, 0, 0] * 3, ring=R)
    assert (Matrix([h]) * V[0][6:, :])[0] == Poly(x ** 2 / 2, t, domain='QQ(x)')
    assert (Matrix([q]) * V[0][:6, :])[0] == Poly(x - S.Half, t, domain='QQ(x)')