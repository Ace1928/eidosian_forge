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
def test_clip_array_min_max(self):
    has_out = (np_clip, np_clip_kwargs, array_clip, array_clip_kwargs)
    has_no_out = (np_clip_no_out, array_clip_no_out)
    a = np.linspace(-10, 10, 40).reshape(5, 2, 4)
    a_min_arr = np.arange(-8, 0).astype(a.dtype).reshape(2, 4)
    a_max_arr = np.arange(0, 8).astype(a.dtype).reshape(2, 4)
    mins = [0, -5, a_min_arr, None]
    maxs = [0, 5, a_max_arr, None]
    for pyfunc in has_out + has_no_out:
        cfunc = jit(nopython=True)(pyfunc)
        for a_min in mins:
            for a_max in maxs:
                if a_min is None and a_max is None:
                    msg = 'array_clip: must set either max or min'
                    with self.assertRaisesRegex(ValueError, msg):
                        cfunc(a, None, None)
                    continue
                np.testing.assert_equal(pyfunc(a, a_min, a_max), cfunc(a, a_min, a_max))
                if pyfunc in has_out:
                    pyout = np.empty_like(a)
                    cout = np.empty_like(a)
                    np.testing.assert_equal(pyfunc(a, a_min, a_max, pyout), cfunc(a, a_min, a_max, cout))
                    np.testing.assert_equal(pyout, cout)
                self._lower_clip_result_test_util(cfunc, a, a_min, a_max)