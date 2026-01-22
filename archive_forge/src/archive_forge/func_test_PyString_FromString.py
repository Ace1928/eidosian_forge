from ctypes import *
import unittest
from test import support
from _ctypes import PyObj_FromPtr
from sys import getrefcount as grc
@support.refcount_test
def test_PyString_FromString(self):
    pythonapi.PyBytes_FromString.restype = py_object
    pythonapi.PyBytes_FromString.argtypes = (c_char_p,)
    s = b'abc'
    refcnt = grc(s)
    pyob = pythonapi.PyBytes_FromString(s)
    self.assertEqual(grc(s), refcnt)
    self.assertEqual(s, pyob)
    del pyob
    self.assertEqual(grc(s), refcnt)