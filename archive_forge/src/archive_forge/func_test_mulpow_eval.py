from sympy.core.function import Function
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, tan)
from sympy.testing.pytest import XFAIL
def test_mulpow_eval():
    x = Symbol('x')
    assert sqrt(50) / (sqrt(2) * x) == 5 / x
    assert sqrt(27) / sqrt(3) == 3