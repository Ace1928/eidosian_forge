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
def check_arr(arr, layout=False):
    np.random.shuffle(_types)
    if layout != False:
        x = np.zeros_like(arr, dtype=_types[0], order=layout)
        y = np.zeros_like(arr, dtype=_types[1], order=layout)
        arr = arr.copy(order=layout)
    else:
        x = np.zeros_like(arr, dtype=_types[0], order=next(layouts))
        y = np.zeros_like(arr, dtype=_types[1], order=next(layouts))
    x.fill(4)
    y.fill(9)
    cfunc = njit((typeof(arr), typeof(x), typeof(y)))(pyfunc)
    expected = pyfunc(arr, x, y)
    got = cfunc(arr, x, y)
    self.assertPreciseEqual(got, expected)