from ctypes import *
from ctypes.test import need_symbol
from test import support
import unittest
import os
import _ctypes_test
def test_unsigned(self):
    for c_typ in unsigned_int_types:

        class X(Structure):
            _fields_ = [('a', c_typ, 3), ('b', c_typ, 3), ('c', c_typ, 1)]
        self.assertEqual(sizeof(X), sizeof(c_typ))
        x = X()
        self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 0, 0, 0))
        x.a = -1
        self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 7, 0, 0))
        x.a, x.b = (0, -1)
        self.assertEqual((c_typ, x.a, x.b, x.c), (c_typ, 0, 7, 0))