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
def test_pow_1():
    assert ((1 + x) ** 2).nseries(x, n=5) == x ** 2 + 2 * x + 1
    assert ((sqrt(x) + 1) ** 2).nseries(x) == 2 * sqrt(x) + x + 1
    assert ((sqrt(x) + cbrt(x)) ** 2).nseries(x) == 2 * x ** Rational(5, 6) + x ** Rational(2, 3) + x