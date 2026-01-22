import unittest
import sys
from ctypes import *
def test_struct_W(self):

    class X(Structure):
        _fields_ = [('a', c_wchar * 3)]
    x = X('abc')
    self.assertRaises(TypeError, X, b'abc')
    self.assertEqual(x.a, 'abc')
    self.assertEqual(type(x.a), str)