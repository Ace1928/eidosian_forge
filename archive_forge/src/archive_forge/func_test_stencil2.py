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
def test_stencil2(self):
    """Tests whether the optional neighborhood argument to the stencil
        decorate works.
        """

    def test_seq(n):
        A = np.arange(n)
        B = stencil2_kernel(A)
        return B

    def test_impl_seq(n):
        A = np.arange(n)
        B = np.zeros(n)
        for i in range(5, len(A)):
            B[i] = 0.3 * sum(A[i - 5:i + 1])
        return B
    n = 100
    self.check(test_impl_seq, test_seq, n)

    def test_seq(n, w):
        A = np.arange(n)

        def stencil2_kernel(a, w):
            cum = a[-w]
            for i in range(-w + 1, w + 1):
                cum += a[i]
            return 0.3 * cum
        B = numba.stencil(stencil2_kernel, neighborhood=((-w, w),))(A, w)
        return B

    def test_impl_seq(n, w):
        A = np.arange(n)
        B = np.zeros(n)
        for i in range(w, len(A) - w):
            B[i] = 0.3 * sum(A[i - w:i + w + 1])
        return B
    n = 100
    w = 5
    cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp))
    expected = test_impl_seq(n, w)
    parfor_output = cpfunc.entry_point(n, w)
    np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
    self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

    def test_seq(n, w, offset):
        A = np.arange(n)

        def stencil2_kernel(a, w):
            cum = a[-w + 1]
            for i in range(-w + 1, w + 1):
                cum += a[i + 1]
            return 0.3 * cum
        B = numba.stencil(stencil2_kernel, neighborhood=((-w, w),), index_offsets=(-offset,))(A, w)
        return B
    offset = 1
    cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp, types.intp))
    parfor_output = cpfunc.entry_point(n, w, offset)
    np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
    self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())

    def test_seq(n, w, offset):
        A = np.arange(n)

        def stencil2_kernel(a, w):
            return 0.3 * np.sum(a[-w + 1:w + 2])
        B = numba.stencil(stencil2_kernel, neighborhood=((-w, w),), index_offsets=(-offset,))(A, w)
        return B
    offset = 1
    cpfunc = self.compile_parallel(test_seq, (types.intp, types.intp, types.intp))
    parfor_output = cpfunc.entry_point(n, w, offset)
    np.testing.assert_almost_equal(parfor_output, expected, decimal=3)
    self.assertIn('@do_scheduling', cpfunc.library.get_llvm_str())