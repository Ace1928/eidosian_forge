from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def check_np_frombuffer(self, pyfunc):
    cfunc = njit(pyfunc)

    def check(buf):
        old_refcnt = sys.getrefcount(buf)
        expected = pyfunc(buf)
        self.memory_leak_setup()
        got = cfunc(buf)
        self.assertPreciseEqual(got, expected)
        del expected
        gc.collect()
        self.assertEqual(sys.getrefcount(buf), old_refcnt + 1)
        del got
        gc.collect()
        self.assertEqual(sys.getrefcount(buf), old_refcnt)
        self.memory_leak_teardown()
    b = bytearray(range(16))
    check(b)
    check(bytes(b))
    check(memoryview(b))
    check(np.arange(12))
    b = np.arange(12).reshape((3, 4))
    check(b)
    self.disable_leak_check()
    with self.assertRaises(ValueError) as raises:
        cfunc(bytearray(b'xxx'))
    self.assertEqual('buffer size must be a multiple of element size', str(raises.exception))