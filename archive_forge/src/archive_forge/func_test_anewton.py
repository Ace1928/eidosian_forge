import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
def test_anewton():
    f = lambda x: (x - 2) ** 100
    x = findroot(f, 1.0, solver=ANewton)
    assert abs(f(x)) < eps