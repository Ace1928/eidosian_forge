from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
def test_array2pointer(self):
    array = (c_int * 3)(42, 17, 2)
    ptr = cast(array, POINTER(c_int))
    self.assertEqual([ptr[i] for i in range(3)], [42, 17, 2])
    if 2 * sizeof(c_short) == sizeof(c_int):
        ptr = cast(array, POINTER(c_short))
        if sys.byteorder == 'little':
            self.assertEqual([ptr[i] for i in range(6)], [42, 0, 17, 0, 2, 0])
        else:
            self.assertEqual([ptr[i] for i in range(6)], [0, 42, 0, 17, 0, 2])