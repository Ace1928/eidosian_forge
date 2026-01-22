import unittest
from ctypes import *
from ctypes.test import need_symbol
def test_checkretval(self):
    import _ctypes_test
    dll = CDLL(_ctypes_test.__file__)
    self.assertEqual(42, dll._testfunc_p_p(42))
    dll._testfunc_p_p.restype = CHECKED
    self.assertEqual('42', dll._testfunc_p_p(42))
    dll._testfunc_p_p.restype = None
    self.assertEqual(None, dll._testfunc_p_p(42))
    del dll._testfunc_p_p.restype
    self.assertEqual(42, dll._testfunc_p_p(42))