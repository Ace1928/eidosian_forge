from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect
from sympy.solvers.ode import (classify_ode,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.utilities.misc import filldedent
def test_2nd_power_series_regular():
    C1, C2, a = symbols('C1 C2 a')
    eq = x ** 2 * f(x).diff(x, 2) - 3 * x * f(x).diff(x) + (4 * x + 4) * f(x)
    sol = Eq(f(x), C1 * x ** 2 * (-16 * x ** 3 / 9 + 4 * x ** 2 - 4 * x + 1) + O(x ** 6))
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = 4 * x ** 2 * f(x).diff(x, 2) - 8 * x ** 2 * f(x).diff(x) + (4 * x ** 2 + 1) * f(x)
    sol = Eq(f(x), C1 * sqrt(x) * (x ** 4 / 24 + x ** 3 / 6 + x ** 2 / 2 + x + 1) + O(x ** 6))
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = x ** 2 * f(x).diff(x, 2) - x ** 2 * f(x).diff(x) + (x ** 2 - 2) * f(x)
    sol = Eq(f(x), C1 * (-x ** 6 / 720 - 3 * x ** 5 / 80 - x ** 4 / 8 + x ** 2 / 2 + x / 2 + 1) / x + C2 * x ** 2 * (-x ** 3 / 60 + x ** 2 / 20 + x / 2 + 1) + O(x ** 6))
    assert dsolve(eq) == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (x ** 2 - Rational(1, 4)) * f(x)
    sol = Eq(f(x), C1 * (x ** 4 / 24 - x ** 2 / 2 + 1) / sqrt(x) + C2 * sqrt(x) * (x ** 4 / 120 - x ** 2 / 6 + 1) + O(x ** 6))
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = x * f(x).diff(x, 2) + f(x).diff(x) - a * x * f(x)
    sol = Eq(f(x), C1 * (a ** 2 * x ** 4 / 64 + a * x ** 2 / 4 + 1) + O(x ** 6))
    assert dsolve(eq, f(x), hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = f(x).diff(x, 2) + (1 - x) / x * f(x).diff(x) + a / x * f(x)
    sol = Eq(f(x), C1 * (-a * x ** 5 * (a - 4) * (a - 3) * (a - 2) * (a - 1) / 14400 + a * x ** 4 * (a - 3) * (a - 2) * (a - 1) / 576 - a * x ** 3 * (a - 2) * (a - 1) / 36 + a * x ** 2 * (a - 1) / 4 - a * x + 1) + O(x ** 6))
    assert dsolve(eq, f(x), hint='2nd_power_series_regular') == sol
    assert checkodesol(eq, sol) == (True, 0)