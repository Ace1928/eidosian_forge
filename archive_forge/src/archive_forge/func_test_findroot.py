import pytest
from mpmath import *
from mpmath.calculus.optimization import Secant, Muller, Bisection, Illinois, \
def test_findroot():
    mp.dps = 15
    assert findroot(lambda x: 4 * x - 3, mpf(5)).ae(0.75)
    assert findroot(sin, mpf(3)).ae(pi)
    assert findroot(sin, (mpf(3), mpf(3.14))).ae(pi)
    assert findroot(lambda x: x * x + 1, mpc(2 + 2j)).ae(1j)
    f = lambda x: cos(x)
    for solver in [Newton, Secant, MNewton, Muller, ANewton]:
        x = findroot(f, 2.0, solver=solver)
        assert abs(f(x)) < eps
    for solver in [Secant, Muller, Bisection, Illinois, Pegasus, Anderson, Ridder]:
        x = findroot(f, (1.0, 2.0), solver=solver)
        assert abs(f(x)) < eps
    f = lambda x: (x - 2) ** 2
    assert isinstance(findroot(f, 1, tol=1e-10), mpf)
    assert isinstance(iv.findroot(f, 1.0, tol=1e-10), iv.mpf)
    assert isinstance(fp.findroot(f, 1, tol=1e-10), float)
    assert isinstance(fp.findroot(f, 1 + 0j, tol=1e-10), complex)
    with pytest.raises(ValueError):
        with workprec(2):
            findroot(lambda x: x ** 2 - 4456178 * x + 60372201703370, mpc(real='5.278e+13', imag='-5.278e+13'))
    with pytest.raises(ValueError):
        findroot(lambda x: -1, 0)
    with pytest.raises(ValueError):
        findroot(lambda p: (1 - p) ** 30 - 1, 0.9)