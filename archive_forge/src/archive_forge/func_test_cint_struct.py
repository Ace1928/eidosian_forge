from ctypes import *
import unittest
def test_cint_struct(self):

    class X(Structure):
        _fields_ = [('a', c_int), ('b', c_int)]
    x = X()
    self.assertEqual(x._objects, None)
    x.a = 42
    x.b = 99
    self.assertEqual(x._objects, None)