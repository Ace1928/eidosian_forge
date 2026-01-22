from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv6():
    x = Symbol('x')
    y = Symbol('y')
    assert (x / 3)._sympy_() == sympy.Symbol('x') / 3
    assert (3 * x)._sympy_() == 3 * sympy.Symbol('x')
    assert (3 + x)._sympy_() == 3 + sympy.Symbol('x')
    assert (3 - x)._sympy_() == 3 - sympy.Symbol('x')
    assert (x / y)._sympy_() == sympy.Symbol('x') / sympy.Symbol('y')