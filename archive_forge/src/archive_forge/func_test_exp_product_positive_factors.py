from sympy.core.evalf import N
from sympy.core.function import (Derivative, Function, PoleError, Subs)
from sympy.core.numbers import (E, Float, Rational, oo, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral, integrate
from sympy.series.order import O
from sympy.series.series import series
from sympy.abc import x, y, n, k
from sympy.testing.pytest import raises
from sympy.series.acceleration import richardson, shanks
from sympy.concrete.summations import Sum
from sympy.core.numbers import Integer
def test_exp_product_positive_factors():
    a, b = symbols('a, b', positive=True)
    x = a * b
    assert series(exp(x), x, n=8) == 1 + a * b + a ** 2 * b ** 2 / 2 + a ** 3 * b ** 3 / 6 + a ** 4 * b ** 4 / 24 + a ** 5 * b ** 5 / 120 + a ** 6 * b ** 6 / 720 + a ** 7 * b ** 7 / 5040 + O(a ** 8 * b ** 8, a, b)