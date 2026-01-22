from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_DifferentialExtension_Rothstein():
    f = (2581284541 * exp(x) + 1757211400) / (39916800 * exp(3 * x) + 119750400 * exp(x) ** 2 + 119750400 * exp(x) + 39916800) * exp(1 / (exp(x) + 1) - 10 * x)
    assert DifferentialExtension(f, x)._important_attrs == (Poly((1757211400 + 2581284541 * t0) * t1, t1), Poly(39916800 + 119750400 * t0 + 119750400 * t0 ** 2 + 39916800 * t0 ** 3, t1), [Poly(1, x), Poly(t0, t0), Poly(-(10 + 21 * t0 + 10 * t0 ** 2) / (1 + 2 * t0 + t0 ** 2) * t1, t1, domain='ZZ(t0)')], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, exp(1 / (t0 + 1) - 10 * i))], [], [None, 'exp', 'exp'], [None, x, 1 / (t0 + 1) - 10 * x])