from symengine.test_utilities import raises
from symengine import (Symbol, Integer, Add, Mul, Pow, Rational, sqrt,
def test_expand2():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert ((1 / (y * z) - y * z) * y * z).expand() == 1 - (y * z) ** 2
    assert (2 * (x + 2 * (y + z))).expand(deep=False) == 2 * x + 4 * (y + z)
    ex = x + 2 * (y + z)
    assert ex.expand(deep=False) == ex