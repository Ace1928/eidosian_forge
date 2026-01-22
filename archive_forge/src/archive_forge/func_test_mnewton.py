import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
def test_mnewton():
    f = lambda x: polyval([1, 3, 3, 1], x)
    x = findroot(f, -0.9, solver='mnewton')
    assert abs(f(x)) < eps