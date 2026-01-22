from ctypes import *
import unittest
from test import support
from _ctypes import PyObj_FromPtr
from sys import getrefcount as grc
@support.refcount_test
def test_PyObj_FromPtr(self):
    s = 'abc def ghi jkl'
    ref = grc(s)
    pyobj = PyObj_FromPtr(id(s))
    self.assertIs(s, pyobj)
    self.assertEqual(grc(s), ref + 1)
    del pyobj
    self.assertEqual(grc(s), ref)