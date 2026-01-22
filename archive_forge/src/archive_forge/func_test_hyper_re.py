from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
from sympy.testing.pytest import raises, XFAIL, slow
def test_hyper_re():
    d = f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == r(k) + (k + 1) * (k + 2) * r(k + 2)
    d = -x * f(x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == (k + 2) * (k + 3) * r(k + 3) - r(k)
    d = 2 * f(x) - 2 * Derivative(f(x), x) + Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == (-2 * k - 2) * r(k + 1) + (k + 1) * (k + 2) * r(k + 2) + 2 * r(k)
    d = 2 * n * f(x) + (x ** 2 - 1) * Derivative(f(x), x)
    assert hyper_re(d, r, k) == k * r(k) + 2 * n * r(k + 1) + (-k - 2) * r(k + 2)
    d = (x ** 10 + 4) * Derivative(f(x), x) + x * (x ** 10 - 1) * Derivative(f(x), x, x)
    assert hyper_re(d, r, k) == (k * (k - 1) + k) * r(k) + (4 * k - (k + 9) * (k + 10) + 40) * r(k + 10)
    d = (x ** 2 - 1) * Derivative(f(x), x, 3) + 3 * x * Derivative(f(x), x, x) + Derivative(f(x), x)
    assert hyper_re(d, r, k) == (k * (k - 2) * (k - 1) + 3 * k * (k - 1) + k) * r(k) + -k * (k + 1) * (k + 2) * r(k + 2)