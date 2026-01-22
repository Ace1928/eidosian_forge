import unittest
from ctypes import *
def test_1_B(self):

    class X(Structure):
        _fields_ = []
    self.assertRaises(AttributeError, setattr, X, '_fields_', [])