from symengine import symbols, sin, sinh, have_numpy, have_llvm, cos, Symbol
from symengine.test_utilities import raises
import pickle
import unittest
def test_pysymbol():
    a = MySymbol('hello', attr=1)
    b = pickle.loads(pickle.dumps(a + 2)) - 2
    try:
        assert a == b
    finally:
        a._unsafe_reset()
        b._unsafe_reset()
    a = MySymbolBase('hello', attr=1)
    try:
        raises(NotImplementedError, lambda: pickle.dumps(a))
        raises(NotImplementedError, lambda: pickle.dumps(a + 2))
    finally:
        a._unsafe_reset()