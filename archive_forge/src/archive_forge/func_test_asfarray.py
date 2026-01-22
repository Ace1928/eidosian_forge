import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_asfarray(self):

    def inputs():
        yield (np.array([1, 2, 3]), None)
        yield (np.array([2, 3], dtype=np.float32), np.float32)
        yield (np.array([2, 3], dtype=np.int8), np.int8)
        yield (np.array([2, 3], dtype=np.int8), np.complex64)
        yield (np.array([2, 3], dtype=np.int8), np.complex128)
    pyfunc = asfarray
    cfunc = jit(nopython=True)(pyfunc)
    for arr, dt in inputs():
        if dt is None:
            expected = pyfunc(arr)
            got = cfunc(arr)
        else:
            expected = pyfunc(arr, dtype=dt)
            got = cfunc(arr, dtype=dt)
        self.assertPreciseEqual(expected, got)
        self.assertTrue(np.issubdtype(got.dtype, np.inexact), got.dtype)
    pyfunc = asfarray_default_kwarg
    cfunc = jit(nopython=True)(pyfunc)
    arr = np.array([1, 2, 3])
    expected = pyfunc(arr)
    got = cfunc(arr)
    self.assertPreciseEqual(expected, got)
    self.assertTrue(np.issubdtype(got.dtype, np.inexact), got.dtype)