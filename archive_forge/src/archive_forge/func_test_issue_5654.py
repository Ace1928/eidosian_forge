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
def test_issue_5654():
    a = Symbol('a')
    assert (1 / (x ** 2 + a ** 2) ** 2).nseries(x, x0=I * a, n=0) == -I / (4 * a ** 3 * (-I * a + x)) - 1 / (4 * a ** 2 * (-I * a + x) ** 2) + O(1, (x, I * a))
    assert (1 / (x ** 2 + a ** 2) ** 2).nseries(x, x0=I * a, n=1) == 3 / (16 * a ** 4) - I / (4 * a ** 3 * (-I * a + x)) - 1 / (4 * a ** 2 * (-I * a + x) ** 2) + O(-I * a + x, (x, I * a))