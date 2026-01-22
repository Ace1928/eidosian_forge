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
def test_nseries():
    assert sqrt(I * x - 1)._eval_nseries(x, 4, None, 1) == I + x / 2 + I * x ** 2 / 8 - x ** 3 / 16 + O(x ** 4)
    assert sqrt(I * x - 1)._eval_nseries(x, 4, None, -1) == -I - x / 2 - I * x ** 2 / 8 + x ** 3 / 16 + O(x ** 4)
    assert cbrt(I * x - 1)._eval_nseries(x, 4, None, 1) == (-1) ** (S(1) / 3) - (-1) ** (S(5) / 6) * x / 3 + (-1) ** (S(1) / 3) * x ** 2 / 9 + 5 * (-1) ** (S(5) / 6) * x ** 3 / 81 + O(x ** 4)
    assert cbrt(I * x - 1)._eval_nseries(x, 4, None, -1) == -(-1) ** (S(2) / 3) - (-1) ** (S(1) / 6) * x / 3 - (-1) ** (S(2) / 3) * x ** 2 / 9 + 5 * (-1) ** (S(1) / 6) * x ** 3 / 81 + O(x ** 4)
    assert (1 / (exp(-1 / x) + 1 / x))._eval_nseries(x, 2, None) == x + O(x ** 2)
    assert sqrt(-I * x ** 2 + x - 3)._eval_nseries(x, 4, None, 1) == -sqrt(3) * I + sqrt(3) * I * x / 6 - sqrt(3) * I * x ** 2 * (-S(1) / 72 + I / 6) - sqrt(3) * I * x ** 3 * (-S(1) / 432 + I / 36) + O(x ** 4)
    assert sqrt(-I * x ** 2 + x - 3)._eval_nseries(x, 4, None, -1) == -sqrt(3) * I + sqrt(3) * I * x / 6 - sqrt(3) * I * x ** 2 * (-S(1) / 72 + I / 6) - sqrt(3) * I * x ** 3 * (-S(1) / 432 + I / 36) + O(x ** 4)
    assert cbrt(-I * x ** 2 + x - 3)._eval_nseries(x, 4, None, 1) == -(-1) ** (S(2) / 3) * 3 ** (S(1) / 3) + (-1) ** (S(2) / 3) * 3 ** (S(1) / 3) * x / 9 - (-1) ** (S(2) / 3) * 3 ** (S(1) / 3) * x ** 2 * (-S(1) / 81 + I / 9) - (-1) ** (S(2) / 3) * 3 ** (S(1) / 3) * x ** 3 * (-S(5) / 2187 + 2 * I / 81) + O(x ** 4)
    assert cbrt(-I * x ** 2 + x - 3)._eval_nseries(x, 4, None, -1) == -(-1) ** (S(2) / 3) * 3 ** (S(1) / 3) + (-1) ** (S(2) / 3) * 3 ** (S(1) / 3) * x / 9 - (-1) ** (S(2) / 3) * 3 ** (S(1) / 3) * x ** 2 * (-S(1) / 81 + I / 9) - (-1) ** (S(2) / 3) * 3 ** (S(1) / 3) * x ** 3 * (-S(5) / 2187 + 2 * I / 81) + O(x ** 4)