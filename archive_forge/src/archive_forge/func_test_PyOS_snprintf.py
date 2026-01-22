from ctypes import *
import unittest
from test import support
from _ctypes import PyObj_FromPtr
from sys import getrefcount as grc
def test_PyOS_snprintf(self):
    PyOS_snprintf = pythonapi.PyOS_snprintf
    PyOS_snprintf.argtypes = (POINTER(c_char), c_size_t, c_char_p)
    buf = c_buffer(256)
    PyOS_snprintf(buf, sizeof(buf), b'Hello from %s', b'ctypes')
    self.assertEqual(buf.value, b'Hello from ctypes')
    PyOS_snprintf(buf, sizeof(buf), b'Hello from %s (%d, %d, %d)', b'ctypes', 1, 2, 3)
    self.assertEqual(buf.value, b'Hello from ctypes (1, 2, 3)')
    self.assertRaises(TypeError, PyOS_snprintf, buf)