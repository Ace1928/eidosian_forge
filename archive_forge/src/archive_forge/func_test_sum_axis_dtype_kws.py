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
def test_sum_axis_dtype_kws(self):
    """ test sum with axis and dtype parameters over a whole range of dtypes """
    pyfunc = array_sum_axis_dtype_kws
    cfunc = jit(nopython=True)(pyfunc)
    all_dtypes = [np.float64, np.float32, np.int64, np.int32, np.uint32, np.uint64, np.complex64, np.complex128]
    all_test_arrays = [[np.ones((7, 6, 5, 4, 3), arr_dtype), np.ones(1, arr_dtype), np.ones((7, 3), arr_dtype) * -5] for arr_dtype in all_dtypes]
    out_dtypes = {np.dtype('float64'): [np.float64], np.dtype('float32'): [np.float64, np.float32], np.dtype('int64'): [np.float64, np.int64, np.float32], np.dtype('int32'): [np.float64, np.int64, np.float32, np.int32], np.dtype('uint32'): [np.float64, np.int64, np.float32], np.dtype('uint64'): [np.float64, np.uint64], np.dtype('complex64'): [np.complex64, np.complex128], np.dtype('complex128'): [np.complex128]}
    for arr_list in all_test_arrays:
        for arr in arr_list:
            for out_dtype in out_dtypes[arr.dtype]:
                for axis in (0, 1, 2):
                    if axis > len(arr.shape) - 1:
                        continue
                    subtest_str = 'Testing np.sum with {} input and {} output '.format(arr.dtype, out_dtype)
                    with self.subTest(subtest_str):
                        py_res = pyfunc(arr, axis=axis, dtype=out_dtype)
                        nb_res = cfunc(arr, axis=axis, dtype=out_dtype)
                        self.assertPreciseEqual(py_res, nb_res)