from symengine import Symbol, symbols, symarray, has_symbol, Dummy
from symengine.test_utilities import raises
import unittest
import platform
def test_has_symbol():
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    assert not has_symbol(2, a)
    assert not has_symbol(c, a)
    assert has_symbol(a + b, b)