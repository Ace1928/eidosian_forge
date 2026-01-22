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
def test_issue_4441():
    a, b = symbols('a,b')
    f = 1 / (1 + a * x)
    assert f.series(x, 0, 5) == 1 - a * x + a ** 2 * x ** 2 - a ** 3 * x ** 3 + a ** 4 * x ** 4 + O(x ** 5)
    f = 1 / (1 + (a + b) * x)
    assert f.series(x, 0, 3) == 1 + x * (-a - b) + x ** 2 * (a + b) ** 2 + O(x ** 3)