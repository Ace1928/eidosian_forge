import unittest
from ctypes import *
from sys import getrefcount as grc
def test_ptr_struct(self):

    class X(Structure):
        _fields_ = [('data', POINTER(c_int))]
    A = c_int * 4
    a = A(11, 22, 33, 44)
    self.assertEqual(a._objects, None)
    x = X()
    x.data = a