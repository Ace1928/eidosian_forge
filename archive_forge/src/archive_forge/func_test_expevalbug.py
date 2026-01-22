from sympy.core.numbers import Rational
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.exponential import exp
def test_expevalbug():
    x = Symbol('x')
    e1 = exp(1 * x)
    e3 = exp(x)
    assert e1 == e3