from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
def test_address2pointer(self):
    array = (c_int * 3)(42, 17, 2)
    address = addressof(array)
    ptr = cast(c_void_p(address), POINTER(c_int))
    self.assertEqual([ptr[i] for i in range(3)], [42, 17, 2])
    ptr = cast(address, POINTER(c_int))
    self.assertEqual([ptr[i] for i in range(3)], [42, 17, 2])