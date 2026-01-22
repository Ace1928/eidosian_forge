import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def test_basic67b(self):
    """basic 2d induced 1D neighborhood"""

    def kernel(a):
        cumul = 0
        for j in range(-10, 1):
            cumul += a[0, j]
        return cumul / (10 * 5)
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    self.check_exceptions(kernel, a, options={'neighborhood': ((-10, 0),)}, expected_exception=[TypingError, ValueError])