import unittest
from ctypes import *
def test_4(self):

    class X(Structure):
        pass

    class Y(X):
        pass
    self.assertRaises(AttributeError, setattr, X, '_fields_', [])
    Y._fields_ = []
    self.assertRaises(AttributeError, setattr, X, '_fields_', [])