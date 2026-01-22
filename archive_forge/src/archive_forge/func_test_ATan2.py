from symengine.sympy_compat import (Integer, Rational, S, Basic, Add, Mul,
from symengine.test_utilities import raises
def test_ATan2():
    x, y = symbols('x y')
    i = atan2(x, y)
    assert isinstance(i, atan2)
    i = atan2(0, 1)
    assert i == 0