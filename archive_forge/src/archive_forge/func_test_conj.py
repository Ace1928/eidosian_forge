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
def test_conj(self):
    for pyfunc in [array_conj, array_conjugate]:
        cfunc = jit(nopython=True)(pyfunc)
        x = np.linspace(-10, 10)
        np.testing.assert_equal(pyfunc(x), cfunc(x))
        x, y = np.meshgrid(x, x)
        z = x + 1j * y
        np.testing.assert_equal(pyfunc(z), cfunc(z))