from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_tuples_lists():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    z = sympy.Symbol('z')
    L = [x, y, z, x * y, z ** y]
    t = (x, y, z, x * y, z ** y)
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    l2 = [x, y, z, x * y, z ** y]
    t2 = (x, y, z, x * y, z ** y)
    assert sympify(L) == l2
    assert sympify(t) == t2
    assert sympify(L) != t2
    assert sympify(t) != l2
    assert L == l2
    assert t == t2
    assert L != t2
    assert t != l2