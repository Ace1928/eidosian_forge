import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
def signed_unsigned_cmp_test(self, comparison_ufunc):
    self.basic_ufunc_test(comparison_ufunc)
    if numpy_support.numpy_version < (1, 25):
        return
    additional_inputs = ((np.int64(-1), np.uint64(0)), (np.int64(-1), np.uint64(1)), (np.int64(0), np.uint64(0)), (np.int64(0), np.uint64(1)), (np.int64(1), np.uint64(0)), (np.int64(1), np.uint64(1)), (np.uint64(0), np.int64(-1)), (np.uint64(0), np.int64(0)), (np.uint64(0), np.int64(1)), (np.uint64(1), np.int64(-1)), (np.uint64(1), np.int64(0)), (np.uint64(1), np.int64(1)), (np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64), np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64)), (np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64), np.array([-1, -1, 0, 0, 1, 1], dtype=np.int64)))
    pyfunc = self._make_ufunc_usecase(comparison_ufunc)
    for a, b in additional_inputs:
        input_types = (typeof(a), typeof(b))
        output_type = types.Array(types.bool_, 1, 'C')
        argtys = input_types + (output_type,)
        cfunc = self._compile(pyfunc, argtys)
        if isinstance(a, np.ndarray):
            result = np.zeros(a.shape, dtype=np.bool_)
        else:
            result = np.zeros(1, dtype=np.bool_)
        expected = np.zeros_like(result)
        pyfunc(a, b, expected)
        cfunc(a, b, result)
        np.testing.assert_equal(expected, result)