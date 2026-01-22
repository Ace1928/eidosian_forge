import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def check_sizing(self, item_size, nmax):
    l = List(self, item_size, 0)

    def make_item(v):
        tmp = '{:0{}}'.format(nmax - v - 1, item_size).encode('latin-1')
        return tmp[:item_size]
    for i in range(nmax):
        l.append(make_item(i))
    self.assertEqual(len(l), nmax)
    for i in range(nmax):
        self.assertEqual(l[i], make_item(i))