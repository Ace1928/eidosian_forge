from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit2():
    x = Symbol('x')
    y = Symbol('y')
    assert x + x == Integer(2) * x
    assert x + x != Integer(3) * x
    assert x + y == y + x
    assert x + x == 2 * x
    assert x + x == x * 2
    assert x + x + x == 3 * x
    assert x + y + x + x == 3 * x + y
    assert not x + x == 3 * x
    assert not x + x != 2 * x