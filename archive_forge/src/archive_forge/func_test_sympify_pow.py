from symengine.test_utilities import raises
from symengine import (Symbol, Integer, sympify, SympifyError, true, false, pi, nan, oo,
from symengine.lib.symengine_wrapper import _sympify, S, One, polygamma
def test_sympify_pow():
    assert sympify('y*pow(x, -1)') == Symbol('y') / Symbol('x')