from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv12b():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    assert sympify(sympy.sinh(x / 3)) == sinh(Symbol('x') / 3)
    assert sympify(sympy.cosh(x / 3)) == cosh(Symbol('x') / 3)
    assert sympify(sympy.tanh(x / 3)) == tanh(Symbol('x') / 3)
    assert sympify(sympy.coth(x / 3)) == coth(Symbol('x') / 3)
    assert sympify(sympy.asinh(x / 3)) == asinh(Symbol('x') / 3)
    assert sympify(sympy.acosh(x / 3)) == acosh(Symbol('x') / 3)
    assert sympify(sympy.atanh(x / 3)) == atanh(Symbol('x') / 3)
    assert sympify(sympy.acoth(x / 3)) == acoth(Symbol('x') / 3)