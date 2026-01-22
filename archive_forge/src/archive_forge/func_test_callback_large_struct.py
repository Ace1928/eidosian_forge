import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_callback_large_struct(self):

    class Check:
        pass

    class X(Structure):
        _fields_ = [('first', c_ulong), ('second', c_ulong), ('third', c_ulong)]

    def callback(check, s):
        check.first = s.first
        check.second = s.second
        check.third = s.third
        s.first = s.second = s.third = 195948557
    check = Check()
    s = X()
    s.first = 3735928559
    s.second = 3405691582
    s.third = 195894762
    CALLBACK = CFUNCTYPE(None, X)
    dll = CDLL(_ctypes_test.__file__)
    func = dll._testfunc_cbk_large_struct
    func.argtypes = (X, CALLBACK)
    func.restype = None
    func(s, CALLBACK(functools.partial(callback, check)))
    self.assertEqual(check.first, s.first)
    self.assertEqual(check.second, s.second)
    self.assertEqual(check.third, s.third)
    self.assertEqual(check.first, 3735928559)
    self.assertEqual(check.second, 3405691582)
    self.assertEqual(check.third, 195894762)
    self.assertEqual(s.first, check.first)
    self.assertEqual(s.second, check.second)
    self.assertEqual(s.third, check.third)