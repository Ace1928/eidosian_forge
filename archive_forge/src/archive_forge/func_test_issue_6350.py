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
def test_issue_6350():
    expr = integrate(exp(k * (y ** 3 - 3 * y)), (y, 0, oo), conds='none')
    assert expr.series(k, 0, 3) == -(-1) ** (S(2) / 3) * sqrt(3) * gamma(S(1) / 3) ** 2 * gamma(S(2) / 3) / (6 * pi * k ** (S(1) / 3)) - sqrt(3) * k * gamma(-S(2) / 3) * gamma(-S(1) / 3) / (6 * pi) - (-1) ** (S(1) / 3) * sqrt(3) * k ** (S(1) / 3) * gamma(-S(1) / 3) * gamma(S(1) / 3) * gamma(S(2) / 3) / (6 * pi) - (-1) ** (S(2) / 3) * sqrt(3) * k ** (S(5) / 3) * gamma(S(1) / 3) ** 2 * gamma(S(2) / 3) / (4 * pi) - (-1) ** (S(1) / 3) * sqrt(3) * k ** (S(7) / 3) * gamma(-S(1) / 3) * gamma(S(1) / 3) * gamma(S(2) / 3) / (8 * pi) + O(k ** 3)