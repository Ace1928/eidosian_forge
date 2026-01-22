import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_allocation(self):
    for i in range(16):
        l = List(self, 8, i)
        self.assertEqual(len(l), 0)
        self.assertEqual(l.allocated, i)