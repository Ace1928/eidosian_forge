import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_sizing(self):
    for i in range(1, 16):
        self.check_sizing(item_size=i, nmax=2 ** i)