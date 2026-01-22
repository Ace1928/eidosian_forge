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
def test_issue_6208():
    from sympy.functions.elementary.miscellaneous import root
    assert sqrt(33 ** (I * 9 / 10)) == -33 ** (I * 9 / 20)
    assert root((6 * I) ** (2 * I), 3).as_base_exp()[1] == Rational(1, 3)
    assert root((6 * I) ** (I / 3), 3).as_base_exp()[1] == I / 9
    assert sqrt(exp(3 * I)) == exp(3 * I / 2)
    assert sqrt(-sqrt(3) * (1 + 2 * I)) == sqrt(sqrt(3)) * sqrt(-1 - 2 * I)
    assert sqrt(exp(5 * I)) == -exp(5 * I / 2)
    assert root(exp(5 * I), 3).exp == Rational(1, 3)