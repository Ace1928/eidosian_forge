from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv5b():
    x = sympy.Integer(5)
    y = sympy.Integer(6)
    assert sympify(x) == Integer(5)
    assert sympify(x / y) == Integer(5) / Integer(6)