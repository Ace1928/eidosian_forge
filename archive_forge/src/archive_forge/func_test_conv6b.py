from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv6b():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    assert sympify(x / 3) == Symbol('x') / 3
    assert sympify(3 * x) == 3 * Symbol('x')
    assert sympify(3 + x) == 3 + Symbol('x')
    assert sympify(3 - x) == 3 - Symbol('x')
    assert sympify(x / y) == Symbol('x') / Symbol('y')