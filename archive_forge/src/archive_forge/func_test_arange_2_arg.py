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
def test_arange_2_arg(self):

    def check_ok(arg0, arg1, pyfunc, cfunc):
        expected = pyfunc(arg0, arg1)
        got = cfunc(arg0, arg1)
        np.testing.assert_allclose(expected, got)
    all_pyfuncs = (np_arange_2, np_arange_start_stop, np_arange_1_stop, np_arange_1_step, lambda x, y: np.arange(x, y, 5), lambda x, y: np.arange(2, y, step=x))
    for pyfunc in all_pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)
        check_ok(-1, 5, pyfunc, cfunc)
        check_ok(-8, -1, pyfunc, cfunc)
        check_ok(4, 0.5, pyfunc, cfunc)
        check_ok(0.5, 4, pyfunc, cfunc)
        check_ok(complex(1, 1), complex(4, 4), pyfunc, cfunc)
        check_ok(complex(4, 4), complex(1, 1), pyfunc, cfunc)
        check_ok(3, None, pyfunc, cfunc)
    pyfunc = np_arange_1_dtype
    cfunc = jit(nopython=True)(pyfunc)
    check_ok(5, np.float32, pyfunc, cfunc)
    check_ok(2.0, np.int32, pyfunc, cfunc)
    check_ok(10, np.complex128, pyfunc, cfunc)
    check_ok(np.complex64(10), np.complex128, pyfunc, cfunc)
    check_ok(7, None, pyfunc, cfunc)
    check_ok(np.int8(0), None, pyfunc, cfunc)