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
def test_recognize_log_derivative():
    a = Poly(2 * x ** 2 + 4 * x * t - 2 * t - x ** 2 * t, t)
    d = Poly((2 * x + t) * (t + x ** 2), t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    assert recognize_log_derivative(a, d, DE, z) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    assert recognize_log_derivative(Poly(t + 1, t), Poly(t + x, t), DE) == True
    assert recognize_log_derivative(Poly(2, t), Poly(t ** 2 - 1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert recognize_log_derivative(Poly(1, x), Poly(x ** 2 - 2, x), DE) == False
    assert recognize_log_derivative(Poly(1, x), Poly(x ** 2 + x, x), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert recognize_log_derivative(Poly(1, t), Poly(t ** 2 - 2, t), DE) == False
    assert recognize_log_derivative(Poly(1, t), Poly(t ** 2 + t, t), DE) == False