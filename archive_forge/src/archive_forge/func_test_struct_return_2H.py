from ctypes import *
from ctypes.test import need_symbol
import sys, unittest
import _ctypes_test
def test_struct_return_2H(self):

    class S2H(Structure):
        _fields_ = [('x', c_short), ('y', c_short)]
    dll.ret_2h_func.restype = S2H
    dll.ret_2h_func.argtypes = [S2H]
    inp = S2H(99, 88)
    s2h = dll.ret_2h_func(inp)
    self.assertEqual((s2h.x, s2h.y), (99 * 2, 88 * 3))