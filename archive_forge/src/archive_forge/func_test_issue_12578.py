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
def test_issue_12578():
    y = (1 - 1 / (x / 2 - 1 / (2 * x)) ** 4) ** (S(1) / 8)
    assert y.series(x, 0, n=17) == 1 - 2 * x ** 4 - 8 * x ** 6 - 34 * x ** 8 - 152 * x ** 10 - 714 * x ** 12 - 3472 * x ** 14 - 17318 * x ** 16 + O(x ** 17)