from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
@unittest.skipIf(not have_sympy, 'SymPy not installed')
def test_mpfr():
    if have_mpfr:
        a = RealMPFR('100', 100)
        b = sympy.Float('100', 29)
        assert sympify(b) == a
        assert b == a._sympy_()