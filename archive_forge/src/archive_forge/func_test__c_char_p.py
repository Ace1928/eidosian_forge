import unittest
from test import support
from ctypes import *
import _ctypes_test
def test__c_char_p(self):

    class X(Structure):
        _fields_ = [('str', c_char_p)]
    x = X()
    self.assertEqual(x.str, None)
    x.str = b'Hello, World'
    self.assertEqual(x.str, b'Hello, World')
    b = c_buffer(b'Hello, World')
    self.assertRaises(TypeError, setattr, x, b'str', b)