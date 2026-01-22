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
def check_scal(scal):
    x = 4
    y = 5
    np.random.shuffle(_types)
    x = _types[0](4)
    y = _types[1](5)
    cfunc = njit((typeof(scal), typeof(x), typeof(y)))(pyfunc)
    expected = pyfunc(scal, x, y)
    got = cfunc(scal, x, y)
    self.assertPreciseEqual(got, expected)