from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x
def test_AccumBounds_func():
    assert (x ** 2 + 2 * x + 1).subs(x, B(-1, 1)) == B(-1, 4)
    assert exp(B(0, 1)) == B(1, E)
    assert exp(B(-oo, oo)) == B(0, oo)
    assert log(B(3, 6)) == B(log(3), log(6))