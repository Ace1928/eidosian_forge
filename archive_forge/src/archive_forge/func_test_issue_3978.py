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
def test_issue_3978():
    f = Function('f')
    assert f(x).series(x, 0, 3, dir='-') == f(0) + x * Subs(Derivative(f(x), x), x, 0) + x ** 2 * Subs(Derivative(f(x), x, x), x, 0) / 2 + O(x ** 3)
    assert f(x).series(x, 0, 3) == f(0) + x * Subs(Derivative(f(x), x), x, 0) + x ** 2 * Subs(Derivative(f(x), x, x), x, 0) / 2 + O(x ** 3)
    assert f(x ** 2).series(x, 0, 3) == f(0) + x ** 2 * Subs(Derivative(f(x), x), x, 0) + O(x ** 3)
    assert f(x ** 2 + 1).series(x, 0, 3) == f(1) + x ** 2 * Subs(Derivative(f(x), x), x, 1) + O(x ** 3)

    class TestF(Function):
        pass
    assert TestF(x).series(x, 0, 3) == TestF(0) + x * Subs(Derivative(TestF(x), x), x, 0) + x ** 2 * Subs(Derivative(TestF(x), x, x), x, 0) / 2 + O(x ** 3)