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
def test_basic34(self):
    """More complex rel index with dependency on addition rel index"""

    def kernel(a):
        g = 4.0 + a[0, 1]
        return g + (a[0, 1] + a[1, 0] + a[0, -1] + np.sin(a[-2, 0]))

    def __kernel(a, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __b in range(1, a.shape[1] - 1):
            for __a in range(2, a.shape[0] - 1):
                g = 4.0 + a[__a + 0, __b + 1]
                __b0[__a, __b] = g + (a[__a + 0, __b + 1] + a[__a + 1, __b + 0] + a[__a + 0, __b + -1] + np.sin(a[__a + -2, __b + 0]))
        return __b0
    a = np.arange(144).reshape(12, 12)
    expected = __kernel(a, None)
    self.check_against_expected(kernel, expected, a)