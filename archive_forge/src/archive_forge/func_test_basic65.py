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
def test_basic65(self):
    """basic induced neighborhood test"""

    def kernel(a):
        cumul = 0
        for i in range(-29, 1):
            cumul += a[i]
        return cumul / 30

    def __kernel(a, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __an in range(29, a.shape[0]):
            cumul = 0
            for i in range(-29, 1):
                cumul += a[__an + i]
            __b0[__an,] = cumul / 30
        return __b0
    a = np.arange(60.0)
    nh = ((-29, 0),)
    expected = __kernel(a, nh)
    self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})