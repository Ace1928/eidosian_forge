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
def test_linear_coeff_match():
    n, d = (z * (2 * x + 3 * f(x) + 5), z * (7 * x + 9 * f(x) + 11))
    rat = n / d
    eq1 = sin(rat) + cos(rat.expand())
    obj1 = LinearCoefficients(eq1)
    eq2 = rat
    obj2 = LinearCoefficients(eq2)
    eq3 = log(sin(rat))
    obj3 = LinearCoefficients(eq3)
    ans = (4, Rational(-13, 3))
    assert obj1._linear_coeff_match(eq1, f(x)) == ans
    assert obj2._linear_coeff_match(eq2, f(x)) == ans
    assert obj3._linear_coeff_match(eq3, f(x)) == ans
    eq4 = 3 * x / f(x)
    obj4 = LinearCoefficients(eq4)
    eq5 = (3 * x + 2) / x
    obj5 = LinearCoefficients(eq5)
    eq6 = (3 * x + 2 * f(x) + 1) / (3 * x + 2 * f(x) + 5)
    obj6 = LinearCoefficients(eq6)
    eq7 = (3 * x + 2 * f(x) + sqrt(2)) / (3 * x + 2 * f(x) + 5)
    obj7 = LinearCoefficients(eq7)
    assert obj4._linear_coeff_match(eq4, f(x)) is None
    assert obj5._linear_coeff_match(eq5, f(x)) is None
    assert obj6._linear_coeff_match(eq6, f(x)) is None
    assert obj7._linear_coeff_match(eq7, f(x)) is None