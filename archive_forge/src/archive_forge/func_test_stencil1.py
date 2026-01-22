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
@skip_unsupported
def test_stencil1(self):
    """Tests whether the optional out argument to stencil calls works.
        """

    def test_with_out(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.zeros(n ** 2).reshape((n, n))
        B = stencil1_kernel(A, out=B)
        return B

    def test_without_out(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = stencil1_kernel(A)
        return B

    def test_impl_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.zeros(n ** 2).reshape((n, n))
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j] + A[i, j - 1] + A[i - 1, j])
        return B
    n = 100
    self.check(test_impl_seq, test_with_out, n)
    self.check(test_impl_seq, test_without_out, n)