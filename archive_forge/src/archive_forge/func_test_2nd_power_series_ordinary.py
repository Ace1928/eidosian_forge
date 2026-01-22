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
@slow
def test_2nd_power_series_ordinary():
    C1, C2 = symbols('C1 C2')
    eq = f(x).diff(x, 2) - x * f(x)
    assert classify_ode(eq) == ('2nd_linear_airy', '2nd_power_series_ordinary')
    sol = Eq(f(x), C2 * (x ** 3 / 6 + 1) + C1 * x * (x ** 3 / 12 + 1) + O(x ** 6))
    assert dsolve(eq, hint='2nd_power_series_ordinary') == sol
    assert checkodesol(eq, sol) == (True, 0)
    sol = Eq(f(x), C2 * ((x + 2) ** 4 / 6 + (x + 2) ** 3 / 6 - (x + 2) ** 2 + 1) + C1 * (x + (x + 2) ** 4 / 12 - (x + 2) ** 3 / 3 + S(2)) + O(x ** 6))
    assert dsolve(eq, hint='2nd_power_series_ordinary', x0=-2) == sol
    sol = Eq(f(x), C2 * x + C1 + O(x ** 2))
    assert dsolve(eq, hint='2nd_power_series_ordinary', n=2) == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = (1 + x ** 2) * f(x).diff(x, 2) + 2 * x * f(x).diff(x) - 2 * f(x)
    assert classify_ode(eq) == ('factorable', '2nd_hypergeometric', '2nd_hypergeometric_Integral', '2nd_power_series_ordinary')
    sol = Eq(f(x), C2 * (-x ** 4 / 3 + x ** 2 + 1) + C1 * x + O(x ** 6))
    assert dsolve(eq, hint='2nd_power_series_ordinary') == sol
    assert checkodesol(eq, sol) == (True, 0)
    eq = f(x).diff(x, 2) + x * f(x).diff(x) + f(x)
    assert classify_ode(eq) == ('factorable', '2nd_power_series_ordinary')
    sol = Eq(f(x), C2 * (x ** 4 / 8 - x ** 2 / 2 + 1) + C1 * x * (-x ** 2 / 3 + 1) + O(x ** 6))
    assert dsolve(eq) == sol
    eq = f(x).diff(x, 2) + f(x).diff(x) - x * f(x)
    assert classify_ode(eq) == ('2nd_power_series_ordinary',)
    sol = Eq(f(x), C2 * (-x ** 4 / 24 + x ** 3 / 6 + 1) + C1 * x * (x ** 3 / 24 + x ** 2 / 6 - x / 2 + 1) + O(x ** 6))
    assert dsolve(eq) == sol
    eq = f(x).diff(x, 2) + x * f(x)
    assert classify_ode(eq) == ('2nd_linear_airy', '2nd_power_series_ordinary')
    sol = Eq(f(x), C2 * (x ** 6 / 180 - x ** 3 / 6 + 1) + C1 * x * (-x ** 3 / 12 + 1) + O(x ** 7))
    assert dsolve(eq, hint='2nd_power_series_ordinary', n=7) == sol
    assert checkodesol(eq, sol) == (True, 0)