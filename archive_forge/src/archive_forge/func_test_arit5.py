from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_arit5():
    x = Symbol('x')
    y = Symbol('y')
    e = (x + y) ** 2
    f = e.expand()
    assert e == (x + y) ** 2
    assert e != x ** 2 + 2 * x * y + y ** 2
    assert isinstance(e, Pow)
    assert f == x ** 2 + 2 * x * y + y ** 2
    assert isinstance(f, Add)