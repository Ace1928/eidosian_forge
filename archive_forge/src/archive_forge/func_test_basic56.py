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
def test_basic56(self):
    """2 args, standard_indexing, added complexity """

    def kernel(a, b):
        s = 1
        acc = 0
        for k in b[0, :]:
            acc += k
        t = 2 - s - 1
        return a[0, 1] + b[0, t] + acc

    def __kernel(a, b, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a, b)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __b in range(0, a.shape[1] - 1):
            for __a in range(0, a.shape[0]):
                s = 1
                acc = 0
                for k in b[0, :]:
                    acc += k
                t = 2 - s - 1
                __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, t] + acc
        return __b0
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(12.0).reshape(3, 4)
    expected = __kernel(a, b, None)
    self.check_against_expected(kernel, expected, a, b, options={'standard_indexing': 'b'})