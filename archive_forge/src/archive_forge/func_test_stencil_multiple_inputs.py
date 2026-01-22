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
def test_stencil_multiple_inputs(self):
    """Tests whether multiple inputs of the same size work.
        """

    def test_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.arange(n ** 2).reshape((n, n))
        C = stencil_multiple_input_kernel(A, B)
        return C

    def test_impl_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.arange(n ** 2).reshape((n, n))
        C = np.zeros(n ** 2).reshape((n, n))
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                C[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j] + A[i, j - 1] + A[i - 1, j] + B[i, j + 1] + B[i + 1, j] + B[i, j - 1] + B[i - 1, j])
        return C
    n = 3
    self.check(test_impl_seq, test_seq, n)

    def test_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.arange(n ** 2).reshape((n, n))
        w = 0.25
        C = stencil_multiple_input_kernel_var(A, B, w)
        return C
    self.check(test_impl_seq, test_seq, n)