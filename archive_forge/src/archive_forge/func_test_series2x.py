from sympy.calculus.util import AccumBounds
from sympy.core.function import (Derivative, PoleError)
from sympy.core.numbers import (E, I, Integer, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, cot, sin, tan)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.abc import x, y, z
from sympy.testing.pytest import raises, XFAIL
def test_series2x():
    assert ((x + 1) ** (-2)).nseries(x, 0, 4) == 1 - 2 * x + 3 * x ** 2 - 4 * x ** 3 + O(x ** 4, x)
    assert ((x + 1) ** (-1)).nseries(x, 0, 4) == 1 - x + x ** 2 - x ** 3 + O(x ** 4, x)
    assert ((x + 1) ** 0).nseries(x, 0, 3) == 1
    assert ((x + 1) ** 1).nseries(x, 0, 3) == 1 + x
    assert ((x + 1) ** 2).nseries(x, 0, 3) == x ** 2 + 2 * x + 1
    assert ((x + 1) ** 3).nseries(x, 0, 3) == 1 + 3 * x + 3 * x ** 2 + O(x ** 3)
    assert (1 / (1 + x)).nseries(x, 0, 4) == 1 - x + x ** 2 - x ** 3 + O(x ** 4, x)
    assert (x + 3 / (1 + 2 * x)).nseries(x, 0, 4) == 3 - 5 * x + 12 * x ** 2 - 24 * x ** 3 + O(x ** 4, x)
    assert ((1 / x + 1) ** 3).nseries(x, 0, 3) == 1 + 3 / x + 3 / x ** 2 + x ** (-3)
    assert (1 / (1 + 1 / x)).nseries(x, 0, 4) == x - x ** 2 + x ** 3 - O(x ** 4, x)
    assert (1 / (1 + 1 / x ** 2)).nseries(x, 0, 6) == x ** 2 - x ** 4 + O(x ** 6, x)