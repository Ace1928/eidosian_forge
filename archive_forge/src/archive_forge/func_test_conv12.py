from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_conv12():
    x = Symbol('x')
    y = Symbol('y')
    assert sinh(x / 3) == sinh(sympy.Symbol('x') / 3)
    assert cosh(x / 3) == cosh(sympy.Symbol('x') / 3)
    assert tanh(x / 3) == tanh(sympy.Symbol('x') / 3)
    assert coth(x / 3) == coth(sympy.Symbol('x') / 3)
    assert asinh(x / 3) == asinh(sympy.Symbol('x') / 3)
    assert acosh(x / 3) == acosh(sympy.Symbol('x') / 3)
    assert atanh(x / 3) == atanh(sympy.Symbol('x') / 3)
    assert acoth(x / 3) == acoth(sympy.Symbol('x') / 3)
    assert sinh(x / 3)._sympy_() == sympy.sinh(sympy.Symbol('x') / 3)
    assert cosh(x / 3)._sympy_() == sympy.cosh(sympy.Symbol('x') / 3)
    assert tanh(x / 3)._sympy_() == sympy.tanh(sympy.Symbol('x') / 3)
    assert coth(x / 3)._sympy_() == sympy.coth(sympy.Symbol('x') / 3)
    assert asinh(x / 3)._sympy_() == sympy.asinh(sympy.Symbol('x') / 3)
    assert acosh(x / 3)._sympy_() == sympy.acosh(sympy.Symbol('x') / 3)
    assert atanh(x / 3)._sympy_() == sympy.atanh(sympy.Symbol('x') / 3)
    assert acoth(x / 3)._sympy_() == sympy.acoth(sympy.Symbol('x') / 3)