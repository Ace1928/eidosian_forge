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
def test_basic81(self):
    """ neighborhood, dimensionally incompatible arrays """

    def kernel(a, b):
        cumul = 0
        for i in range(-3, 1):
            for j in range(-3, 1):
                cumul += a[i, j] + b[i]
        return cumul / 9.0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    b = a[0].copy()
    ex = self.exception_dict(stencil=TypingError, parfor=AssertionError, njit=TypingError)
    self.check_exceptions(kernel, a, b, options={'neighborhood': ((-3, 0), (-3, 0))}, expected_exception=ex)