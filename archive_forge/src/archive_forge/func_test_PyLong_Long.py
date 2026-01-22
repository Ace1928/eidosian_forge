from ctypes import *
import unittest
from test import support
from _ctypes import PyObj_FromPtr
from sys import getrefcount as grc
@support.refcount_test
def test_PyLong_Long(self):
    ref42 = grc(42)
    pythonapi.PyLong_FromLong.restype = py_object
    self.assertEqual(pythonapi.PyLong_FromLong(42), 42)
    self.assertEqual(grc(42), ref42)
    pythonapi.PyLong_AsLong.argtypes = (py_object,)
    pythonapi.PyLong_AsLong.restype = c_long
    res = pythonapi.PyLong_AsLong(42)
    self.assertEqual(grc(res), ref42 + 1)
    del res
    self.assertEqual(grc(42), ref42)