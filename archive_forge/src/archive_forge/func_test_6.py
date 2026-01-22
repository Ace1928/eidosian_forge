import unittest
from ctypes import *
def test_6(self):

    class X(Structure):
        _fields_ = [('x', c_int)]
    CField = type(X.x)
    self.assertRaises(TypeError, CField)