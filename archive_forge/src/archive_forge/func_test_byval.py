from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_byval(self):
    ptin = POINT(1, 2)
    ptout = POINT()
    result = dll._testfunc_byval(ptin, byref(ptout))
    got = (result, ptout.x, ptout.y)
    expected = (3, 1, 2)
    self.assertEqual(got, expected)
    ptin = POINT(101, 102)
    ptout = POINT()
    dll._testfunc_byval.argtypes = (POINT, POINTER(POINT))
    dll._testfunc_byval.restype = c_int
    result = dll._testfunc_byval(ptin, byref(ptout))
    got = (result, ptout.x, ptout.y)
    expected = (203, 101, 102)
    self.assertEqual(got, expected)