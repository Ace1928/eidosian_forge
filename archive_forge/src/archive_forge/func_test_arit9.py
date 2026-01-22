from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit9():
    x = Symbol('x')
    y = Symbol('y')
    assert 1 / x == 1 / x
    assert 1 / x != 1 / y