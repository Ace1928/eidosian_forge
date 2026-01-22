from sympy.core import (
from sympy.core.parameters import global_parameters
from sympy.core.tests.test_evalf import NS
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.error_functions import erf
from sympy.functions.elementary.trigonometric import (
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.polys import Poly
from sympy.series.order import O
from sympy.sets import FiniteSet
from sympy.core.power import power, integer_nthroot
from sympy.testing.pytest import warns, _both_exp_pow
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.abc import a, b, c, x, y
def test_issue_6068():
    assert sqrt(sin(x)).series(x, 0, 7) == sqrt(x) - x ** Rational(5, 2) / 12 + x ** Rational(9, 2) / 1440 - x ** Rational(13, 2) / 24192 + O(x ** 7)
    assert sqrt(sin(x)).series(x, 0, 9) == sqrt(x) - x ** Rational(5, 2) / 12 + x ** Rational(9, 2) / 1440 - x ** Rational(13, 2) / 24192 - 67 * x ** Rational(17, 2) / 29030400 + O(x ** 9)
    assert sqrt(sin(x ** 3)).series(x, 0, 19) == x ** Rational(3, 2) - x ** Rational(15, 2) / 12 + x ** Rational(27, 2) / 1440 + O(x ** 19)
    assert sqrt(sin(x ** 3)).series(x, 0, 20) == x ** Rational(3, 2) - x ** Rational(15, 2) / 12 + x ** Rational(27, 2) / 1440 - x ** Rational(39, 2) / 24192 + O(x ** 20)