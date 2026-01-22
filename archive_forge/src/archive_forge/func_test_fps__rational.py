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
@slow
def test_fps__rational():
    assert fps(1 / x) == 1 / x
    assert fps((x ** 2 + x + 1) / x ** 3, dir=-1) == (x ** 2 + x + 1) / x ** 3
    f = 1 / ((x - 1) ** 2 * (x - 2))
    assert fps(f, x).truncate() == Rational(-1, 2) - x * Rational(5, 4) - 17 * x ** 2 / 8 - 49 * x ** 3 / 16 - 129 * x ** 4 / 32 - 321 * x ** 5 / 64 + O(x ** 6)
    f = (1 + x + x ** 2 + x ** 3) / ((x - 1) * (x - 2))
    assert fps(f, x).truncate() == S.Half + x * Rational(5, 4) + 17 * x ** 2 / 8 + 49 * x ** 3 / 16 + 113 * x ** 4 / 32 + 241 * x ** 5 / 64 + O(x ** 6)
    f = x / (1 - x - x ** 2)
    assert fps(f, x, full=True).truncate() == x + x ** 2 + 2 * x ** 3 + 3 * x ** 4 + 5 * x ** 5 + O(x ** 6)
    f = 1 / (x ** 2 + 2 * x + 2)
    assert fps(f, x, full=True).truncate() == S.Half - x / 2 + x ** 2 / 4 - x ** 4 / 8 + x ** 5 / 8 + O(x ** 6)
    f = log(1 + x)
    assert fps(f, x).truncate() == x - x ** 2 / 2 + x ** 3 / 3 - x ** 4 / 4 + x ** 5 / 5 + O(x ** 6)
    assert fps(f, x, dir=1).truncate() == fps(f, x, dir=-1).truncate()
    assert fps(f, x, 2).truncate() == log(3) - Rational(2, 3) - (x - 2) ** 2 / 18 + (x - 2) ** 3 / 81 - (x - 2) ** 4 / 324 + (x - 2) ** 5 / 1215 + x / 3 + O((x - 2) ** 6, (x, 2))
    assert fps(f, x, 2, dir=-1).truncate() == log(3) - Rational(2, 3) - (-x + 2) ** 2 / 18 - (-x + 2) ** 3 / 81 - (-x + 2) ** 4 / 324 - (-x + 2) ** 5 / 1215 + x / 3 + O((x - 2) ** 6, (x, 2))
    f = atan(x)
    assert fps(f, x, full=True).truncate() == x - x ** 3 / 3 + x ** 5 / 5 + O(x ** 6)
    assert fps(f, x, full=True, dir=1).truncate() == fps(f, x, full=True, dir=-1).truncate()
    assert fps(f, x, 2, full=True).truncate() == atan(2) - Rational(2, 5) - 2 * (x - 2) ** 2 / 25 + 11 * (x - 2) ** 3 / 375 - 6 * (x - 2) ** 4 / 625 + 41 * (x - 2) ** 5 / 15625 + x / 5 + O((x - 2) ** 6, (x, 2))
    assert fps(f, x, 2, full=True, dir=-1).truncate() == atan(2) - Rational(2, 5) - 2 * (-x + 2) ** 2 / 25 - 11 * (-x + 2) ** 3 / 375 - 6 * (-x + 2) ** 4 / 625 - 41 * (-x + 2) ** 5 / 15625 + x / 5 + O((x - 2) ** 6, (x, 2))
    f = x * atan(x) - log(1 + x ** 2) / 2
    assert fps(f, x, full=True).truncate() == x ** 2 / 2 - x ** 4 / 12 + O(x ** 6)
    f = log((1 + x) / (1 - x)) / 2 - atan(x)
    assert fps(f, x, full=True).truncate(n=10) == 2 * x ** 3 / 3 + 2 * x ** 7 / 7 + O(x ** 10)