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
def test_issue_14885():
    assert series(x ** Rational(-3, 2) * exp(x), x, 0) == x ** Rational(-3, 2) + 1 / sqrt(x) + sqrt(x) / 2 + x ** Rational(3, 2) / 6 + x ** Rational(5, 2) / 24 + x ** Rational(7, 2) / 120 + x ** Rational(9, 2) / 720 + x ** Rational(11, 2) / 5040 + O(x ** 6)