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
def check_round_array(self, pyfunc):

    def check_round(cfunc, values, inty, outty, decimals):
        arr = values.astype(as_dtype(inty))
        out = np.zeros_like(arr).astype(as_dtype(outty))
        pyout = out.copy()
        _fixed_np_round(arr, decimals, pyout)
        self.memory_leak_setup()
        cfunc(arr, decimals, out)
        self.memory_leak_teardown()
        np.testing.assert_allclose(out, pyout)
        with self.assertRaises(ValueError) as raises:
            cfunc(arr, decimals, out[1:])
        self.assertEqual(str(raises.exception), 'invalid output shape')

    def check_types(argtypes, outtypes, values):
        for inty, outty in product(argtypes, outtypes):
            argtys = (types.Array(inty, 1, 'A'), types.int32, types.Array(outty, 1, 'A'))
            cfunc = njit(argtys)(pyfunc)
            check_round(cfunc, values, inty, outty, 0)
            check_round(cfunc, values, inty, outty, 1)
            if not isinstance(outty, types.Integer):
                check_round(cfunc, values * 10, inty, outty, -1)
            else:
                pass
    values = np.array([-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75])
    argtypes = (types.float64, types.float32)
    check_types(argtypes, argtypes, values)
    argtypes = (types.complex64, types.complex128)
    check_types(argtypes, argtypes, values * (1 - 1j))
    self.disable_leak_check()