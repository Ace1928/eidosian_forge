from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
@unittest.skipUnless(sys.maxsize >= 2 ** 32, 'need a 64-bit system to test for MemoryError')
def test_mul_error(self):
    self.disable_leak_check()
    pyfunc = list_mul
    cfunc = jit(nopython=True)(pyfunc)
    with self.assertRaises(MemoryError):
        cfunc(1, 2 ** 58)
    if sys.platform.startswith('darwin'):
        libc = ct.CDLL('libc.dylib')
        libc.printf("###Please ignore the above error message i.e. can't allocate region. It is in fact the purpose of this test to request more memory than can be provided###\n".encode('UTF-8'))
    with self.assertRaises(MemoryError):
        cfunc(1, 2 ** 62)