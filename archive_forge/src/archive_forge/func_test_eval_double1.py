from symengine.test_utilities import raises
from symengine import (Symbol, sin, cos, Integer, Add, I, RealDouble, ComplexDouble, sqrt)
from unittest.case import SkipTest
def test_eval_double1():
    x = Symbol('x')
    y = Symbol('y')
    e = sin(x) ** 2 + cos(x) ** 2
    e = e.subs(x, 7)
    assert abs(e.n(real=True) - 1) < 1e-09
    assert abs(e.n() - 1) < 1e-09