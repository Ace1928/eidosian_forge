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
def test_fps__fractional():
    f = sin(sqrt(x)) / x
    assert fps(f, x).truncate() == 1 / sqrt(x) - sqrt(x) / 6 + x ** Rational(3, 2) / 120 - x ** Rational(5, 2) / 5040 + x ** Rational(7, 2) / 362880 - x ** Rational(9, 2) / 39916800 + x ** Rational(11, 2) / 6227020800 + O(x ** 6)
    f = sin(sqrt(x)) * x
    assert fps(f, x).truncate() == x ** Rational(3, 2) - x ** Rational(5, 2) / 6 + x ** Rational(7, 2) / 120 - x ** Rational(9, 2) / 5040 + x ** Rational(11, 2) / 362880 + O(x ** 6)
    f = atan(sqrt(x)) / x ** 2
    assert fps(f, x).truncate() == x ** Rational(-3, 2) - x ** Rational(-1, 2) / 3 + x ** S.Half / 5 - x ** Rational(3, 2) / 7 + x ** Rational(5, 2) / 9 - x ** Rational(7, 2) / 11 + x ** Rational(9, 2) / 13 - x ** Rational(11, 2) / 15 + O(x ** 6)
    f = exp(sqrt(x))
    assert fps(f, x).truncate().expand() == 1 + x / 2 + x ** 2 / 24 + x ** 3 / 720 + x ** 4 / 40320 + x ** 5 / 3628800 + sqrt(x) + x ** Rational(3, 2) / 6 + x ** Rational(5, 2) / 120 + x ** Rational(7, 2) / 5040 + x ** Rational(9, 2) / 362880 + x ** Rational(11, 2) / 39916800 + O(x ** 6)
    f = exp(sqrt(x)) * x
    assert fps(f, x).truncate().expand() == x + x ** 2 / 2 + x ** 3 / 24 + x ** 4 / 720 + x ** 5 / 40320 + x ** Rational(3, 2) + x ** Rational(5, 2) / 6 + x ** Rational(7, 2) / 120 + x ** Rational(9, 2) / 5040 + x ** Rational(11, 2) / 362880 + O(x ** 6)