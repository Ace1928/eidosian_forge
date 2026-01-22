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
def test_fps_shift():
    f = x ** (-5) * sin(x)
    assert fps(f, x).truncate() == 1 / x ** 4 - 1 / (6 * x ** 2) + Rational(1, 120) - x ** 2 / 5040 + x ** 4 / 362880 + O(x ** 6)
    f = x ** 2 * atan(x)
    assert fps(f, x, rational=False).truncate() == x ** 3 - x ** 5 / 3 + O(x ** 6)
    f = cos(sqrt(x)) * x
    assert fps(f, x).truncate() == x - x ** 2 / 2 + x ** 3 / 24 - x ** 4 / 720 + x ** 5 / 40320 + O(x ** 6)
    f = x ** 2 * cos(sqrt(x))
    assert fps(f, x).truncate() == x ** 2 - x ** 3 / 2 + x ** 4 / 24 - x ** 5 / 720 + O(x ** 6)