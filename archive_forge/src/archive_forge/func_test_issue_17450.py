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
def test_issue_17450():
    assert (erf(cosh(1) ** 7) ** I).is_real is None
    assert (erf(cosh(1) ** 7) ** I).is_imaginary is False
    assert Pow(exp(1 + sqrt(2)), (1 - sqrt(2)) * I * pi, evaluate=False).is_real is None
    assert ((-10) ** (10 * I * pi / 3)).is_real is False
    assert ((-5) ** (4 * I * pi)).is_real is False