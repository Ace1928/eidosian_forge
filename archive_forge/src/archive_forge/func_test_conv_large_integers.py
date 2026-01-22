from symengine import (Symbol, Integer, sympify, SympifyError, log,
from symengine.lib.symengine_wrapper import (Subs, Derivative, RealMPFR,
import unittest
def test_conv_large_integers():
    a = Integer(10) ** 10000
    b = int(a)
    if have_sympy:
        c = a._sympy_()
        d = sympify(c)