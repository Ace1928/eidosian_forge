import unittest
from test import support
from ctypes import *
import _ctypes_test
@support.refcount_test
def test__POINTER_c_char(self):

    class X(Structure):
        _fields_ = [('str', POINTER(c_char))]
    x = X()
    self.assertRaises(ValueError, getattr, x.str, 'contents')
    b = c_buffer(b'Hello, World')
    from sys import getrefcount as grc
    self.assertEqual(grc(b), 2)
    x.str = b
    self.assertEqual(grc(b), 3)
    for i in range(len(b)):
        self.assertEqual(b[i], x.str[i])
    self.assertRaises(TypeError, setattr, x, 'str', 'Hello, World')