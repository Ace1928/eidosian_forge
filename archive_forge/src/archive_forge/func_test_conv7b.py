from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv7b():
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    assert sympify(sympy.sin(x / 3)) == sin(Symbol('x') / 3)
    assert sympify(sympy.sin(x / 3)) != cos(Symbol('x') / 3)
    assert sympify(sympy.cos(x / 3)) == cos(Symbol('x') / 3)
    assert sympify(sympy.tan(x / 3)) == tan(Symbol('x') / 3)
    assert sympify(sympy.cot(x / 3)) == cot(Symbol('x') / 3)
    assert sympify(sympy.csc(x / 3)) == csc(Symbol('x') / 3)
    assert sympify(sympy.sec(x / 3)) == sec(Symbol('x') / 3)
    assert sympify(sympy.asin(x / 3)) == asin(Symbol('x') / 3)
    assert sympify(sympy.acos(x / 3)) == acos(Symbol('x') / 3)
    assert sympify(sympy.atan(x / 3)) == atan(Symbol('x') / 3)
    assert sympify(sympy.acot(x / 3)) == acot(Symbol('x') / 3)
    assert sympify(sympy.acsc(x / 3)) == acsc(Symbol('x') / 3)
    assert sympify(sympy.asec(x / 3)) == asec(Symbol('x') / 3)
    assert sympify(sympy.atan2(x / 3, y)) == atan2(Symbol('x') / 3, Symbol('y'))