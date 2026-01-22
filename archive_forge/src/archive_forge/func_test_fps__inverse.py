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
def test_fps__inverse():
    f1, f2, f3 = (fps(sin(x)), fps(exp(x)), fps(cos(x)))
    raises(ValueError, lambda: f1.inverse(x))
    finv = f2.inverse(x)
    assert isinstance(finv, FormalPowerSeriesInverse)
    assert isinstance(finv.ffps, FormalPowerSeries)
    raises(ValueError, lambda: finv.gfps)
    assert finv.f == exp(x)
    assert finv.function == exp(-x)
    assert finv._eval_terms(5) == 1 - x + x ** 2 / 2 - x ** 3 / 6 + x ** 4 / 24
    assert finv.truncate() == 1 - x + x ** 2 / 2 - x ** 3 / 6 + x ** 4 / 24 - x ** 5 / 120 + O(x ** 6)
    assert finv.truncate(5) == 1 - x + x ** 2 / 2 - x ** 3 / 6 + x ** 4 / 24 + O(x ** 5)
    raises(NotImplementedError, lambda: finv._eval_term(5))
    raises(ValueError, lambda: finv.g)
    raises(NotImplementedError, lambda: finv.infinite)
    raises(NotImplementedError, lambda: finv._eval_derivative(x))
    raises(NotImplementedError, lambda: finv.integrate(x))
    assert f2.inverse(x).truncate(8) == 1 - x + x ** 2 / 2 - x ** 3 / 6 + x ** 4 / 24 - x ** 5 / 120 + x ** 6 / 720 - x ** 7 / 5040 + O(x ** 8)
    assert f3.inverse(x).truncate() == 1 + x ** 2 / 2 + 5 * x ** 4 / 24 + O(x ** 6)
    assert f3.inverse(x).truncate(8) == 1 + x ** 2 / 2 + 5 * x ** 4 / 24 + 61 * x ** 6 / 720 + O(x ** 8)