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
def test_fps__asymptotic():
    f = exp(x)
    assert fps(f, x, oo) == f
    assert fps(f, x, -oo).truncate() == O(1 / x ** 6, (x, oo))
    f = erf(x)
    assert fps(f, x, oo).truncate() == 1 + O(1 / x ** 6, (x, oo))
    assert fps(f, x, -oo).truncate() == -1 + O(1 / x ** 6, (x, oo))
    f = atan(x)
    assert fps(f, x, oo, full=True).truncate() == -1 / (5 * x ** 5) + 1 / (3 * x ** 3) - 1 / x + pi / 2 + O(1 / x ** 6, (x, oo))
    assert fps(f, x, -oo, full=True).truncate() == -1 / (5 * x ** 5) + 1 / (3 * x ** 3) - 1 / x - pi / 2 + O(1 / x ** 6, (x, oo))
    f = log(1 + x)
    assert fps(f, x, oo) != -1 / (5 * x ** 5) - 1 / (4 * x ** 4) + 1 / (3 * x ** 3) - 1 / (2 * x ** 2) + 1 / x - log(1 / x) + O(1 / x ** 6, (x, oo))
    assert fps(f, x, -oo) != -1 / (5 * x ** 5) - 1 / (4 * x ** 4) + 1 / (3 * x ** 3) - 1 / (2 * x ** 2) + 1 / x + I * pi - log(-1 / x) + O(1 / x ** 6, (x, oo))