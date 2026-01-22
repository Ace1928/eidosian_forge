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
def test_basic47(self):
    """3 args"""

    def kernel(a, b, c):
        return a[0, 1] + b[1, 0] + c[-1, 0]

    def __kernel(a, b, c, neighborhood):
        self.check_stencil_arrays(a, b, c, neighborhood=neighborhood)
        __retdtype = kernel(a, b, c)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __b in range(0, a.shape[1] - 1):
            for __a in range(1, a.shape[0] - 1):
                __b0[__a, __b] = a[__a + 0, __b + 1] + b[__a + 1, __b + 0] + c[__a + -1, __b + 0]
        return __b0
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(12.0).reshape(3, 4)
    c = np.arange(12.0).reshape(3, 4)
    expected = __kernel(a, b, c, None)
    self.check_against_expected(kernel, expected, a, b, c)