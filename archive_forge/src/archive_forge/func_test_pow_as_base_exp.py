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
def test_pow_as_base_exp():
    assert (S.Infinity ** (2 - x)).as_base_exp() == (S.Infinity, 2 - x)
    assert (S.Infinity ** (x - 2)).as_base_exp() == (S.Infinity, x - 2)
    p = S.Half ** x
    assert p.base, p.exp == p.as_base_exp() == (S(2), -x)
    p = (S(3) / 2) ** x
    assert p.base, p.exp == p.as_base_exp() == (3 * S.Half, x)
    p = (S(2) / 3) ** x
    assert p.as_base_exp() == (S(3) / 2, -x)
    assert p.base, p.exp == (S(2) / 3, x)
    assert Pow(1, 2, evaluate=False).as_base_exp() == (S.One, S(2))