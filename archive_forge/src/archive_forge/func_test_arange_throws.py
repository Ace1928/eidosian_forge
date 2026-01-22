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
def test_arange_throws(self):
    self.disable_leak_check()
    bad_funcs_1 = [lambda x: np.arange(stop=x), lambda x: np.arange(step=x), lambda x: np.arange(dtype=x)]
    bad_funcs_2 = [lambda x, y: np.arange(stop=x, step=y), lambda x, y: np.arange(stop=x, dtype=y)]
    for pyfunc in bad_funcs_1:
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(pyfunc)
            cfunc(2)
    for pyfunc in bad_funcs_2:
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(pyfunc)
            cfunc(2, 6)
    pyfunc = np_arange_3
    cfunc = jit(nopython=True)(pyfunc)
    for f in (pyfunc, cfunc):
        for inputs in [(1, np.int16(2), 0), (1, 2, 0)]:
            permitted_errors = (ZeroDivisionError, ValueError)
            with self.assertRaises(permitted_errors) as raises:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    f(*inputs)
                self.assertIn('Maximum allowed size exceeded', str(raises.exception))