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
def test_powers_of_I():
    assert [sqrt(I) ** i for i in range(13)] == [1, sqrt(I), I, sqrt(I) ** 3, -1, -sqrt(I), -I, -sqrt(I) ** 3, 1, sqrt(I), I, sqrt(I) ** 3, -1]
    assert sqrt(I) ** (S(9) / 2) == -I ** (S(1) / 4)