from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv8b():
    e1 = sympy.Function('f')(sympy.Symbol('x'))
    e2 = sympy.Function('g')(sympy.Symbol('x'), sympy.Symbol('y'))
    assert sympify(e1) == function_symbol('f', Symbol('x'))
    assert sympify(e2) != function_symbol('f', Symbol('x'))
    assert sympify(e2) == function_symbol('g', Symbol('x'), Symbol('y'))
    e3 = sympy.Function('q')(sympy.Symbol('t'))
    assert sympify(e3) == function_symbol('q', Symbol('t'))
    assert sympify(e3) != function_symbol('f', Symbol('t'))
    assert sympify(e3) != function_symbol('q', Symbol('t'), Symbol('t'))