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
def test_recognize_derivative():
    DE = DifferentialExtension(extension={'D': [Poly(1, t)]})
    a = Poly(36, t)
    d = Poly((t - 2) * (t ** 2 - 1) ** 2, t)
    assert recognize_derivative(a, d, DE) == False
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 / x, t)]})
    a = Poly(2, t)
    d = Poly(t ** 2 - 1, t)
    assert recognize_derivative(a, d, DE) == False
    assert recognize_derivative(Poly(x * t, t), Poly(1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t ** 2 + 1, t)]})
    assert recognize_derivative(Poly(t, t), Poly(1, t), DE) == True