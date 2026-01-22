import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
@skip_parfors_unsupported
def test_array_ctypes_ref_error_in_parallel(self):
    from ctypes import CFUNCTYPE, c_void_p, c_int32, c_double, c_bool

    @CFUNCTYPE(c_bool, c_void_p, c_int32, c_void_p)
    def callback(inptr, size, outptr):
        try:
            inbuf = (c_double * size).from_address(inptr)
            outbuf = (c_double * 1).from_address(outptr)
            a = np.ndarray(size, buffer=inbuf, dtype=np.float64)
            b = np.ndarray(1, buffer=outbuf, dtype=np.float64)
            b[0] = (a + a.size)[0]
            return True
        except:
            import traceback
            traceback.print_exception()
            return False

    @njit(parallel=True)
    def foo(size):
        arr = np.ones(size)
        out = np.empty(1)
        inct = arr.ctypes
        outct = out.ctypes
        status = callback(inct.data, size, outct.data)
        return (status, out[0])
    size = 3
    status, got = foo(size)
    self.assertTrue(status)
    self.assertPreciseEqual(got, (np.ones(size) + size)[0])