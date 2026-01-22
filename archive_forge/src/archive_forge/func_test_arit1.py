from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit1():
    x = Symbol('x')
    y = Symbol('y')
    e = x + y
    e = x * y
    e = Integer(2) * x
    e = 2 * x
    e = x + 1
    e = 1 + x