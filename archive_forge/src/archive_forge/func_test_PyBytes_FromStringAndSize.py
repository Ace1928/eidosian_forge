from ctypes import *
import unittest
from test import support
from _ctypes import PyObj_FromPtr
from sys import getrefcount as grc
def test_PyBytes_FromStringAndSize(self):
    PyBytes_FromStringAndSize = pythonapi.PyBytes_FromStringAndSize
    PyBytes_FromStringAndSize.restype = py_object
    PyBytes_FromStringAndSize.argtypes = (c_char_p, c_size_t)
    self.assertEqual(PyBytes_FromStringAndSize(b'abcdefghi', 3), b'abc')