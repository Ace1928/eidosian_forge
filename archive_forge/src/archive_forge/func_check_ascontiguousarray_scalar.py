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
def check_ascontiguousarray_scalar(self, pyfunc):

    def check_scalar(x):
        cfunc = njit((typeof(x),))(pyfunc)
        expected = pyfunc(x)
        got = cfunc(x)
        self.assertPreciseEqual(expected, got)
    for x in [42, 42.0, 42j, np.float32(42), np.float64(42), True]:
        check_scalar(x)