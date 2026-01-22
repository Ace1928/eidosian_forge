from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
def positive_address(a):
    if a >= 0:
        return a
    import struct
    num_bits = struct.calcsize('P') * 8
    a += 1 << num_bits
    assert a >= 0
    return a