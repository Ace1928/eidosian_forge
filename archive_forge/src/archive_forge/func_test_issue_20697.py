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
def test_issue_20697():
    p_0, p_1, p_2, p_3, b_0, b_1, b_2 = symbols('p_0 p_1 p_2 p_3 b_0 b_1 b_2')
    Q = (p_0 + (p_1 + (p_2 + p_3 / y) / y) / y) / (1 + ((p_3 / (b_0 * y) + (b_0 * p_2 - b_1 * p_3) / b_0 ** 2) / y + (b_0 ** 2 * p_1 - b_0 * b_1 * p_2 - p_3 * (b_0 * b_2 - b_1 ** 2)) / b_0 ** 3) / y)
    assert Q.series(y, n=3).ratsimp() == b_2 * y ** 2 + b_1 * y + b_0 + O(y ** 3)