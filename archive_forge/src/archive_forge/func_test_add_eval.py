from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL
def test_add_eval():
    a = Symbol('a')
    b = Symbol('b')
    c = Rational(1)
    p = Rational(5)
    assert a * b + c + p == a * b + 6
    assert c + a + p == a + 6
    assert c + a - p == a + -4
    assert a + a == 2 * a
    assert a + p + a == 2 * a + 5
    assert c + p == Rational(6)
    assert b + a - b == a