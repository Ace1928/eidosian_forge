from symengine import (
from symengine.test_utilities import raises
import unittest
@unittest.skipUnless(have_sympy, 'SymPy not installed')
def test_FunctionWrapper():
    import sympy
    n, m, theta, phi = sympy.symbols('n, m, theta, phi')
    r = sympy.Ynm(n, m, theta, phi)
    s = Integer(2) * r
    assert isinstance(s, Mul)
    assert isinstance(s.args[1]._sympy_(), sympy.Ynm)
    x = symbols('x')
    e = x + sympy.Mod(x, 2)
    assert str(e) == 'x + Mod(x, 2)'
    assert isinstance(e, Add)
    assert e + sympy.Mod(x, 2) == x + 2 * sympy.Mod(x, 2)
    f = e.subs({x: 10})
    assert f == 10
    f = e.subs({x: 2})
    assert f == 2
    f = e.subs({x: 100})
    v = f.n(53, real=True)
    assert abs(float(v) - 100.0) < 1e-07