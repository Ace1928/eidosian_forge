import unittest
from ctypes import *
from sys import getrefcount as grc
def test_ints(self):
    i = 42000123
    refcnt = grc(i)
    ci = c_int(i)
    self.assertEqual(refcnt, grc(i))
    self.assertEqual(ci._objects, None)