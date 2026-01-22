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
def test_stencil_call_const(self):
    """Tests numba.stencil call that has an index that can be inferred as
        constant from a unary expr. Otherwise, this would raise an error since
        neighborhood length is not specified.
        """

    def test_impl1(n):
        A = np.arange(n)
        B = np.zeros(n)
        c = 1
        numba.stencil(lambda a, c: 0.3 * (a[-c] + a[0] + a[c]))(A, c, out=B)
        return B

    def test_impl2(n):
        A = np.arange(n)
        B = np.zeros(n)
        c = 2
        numba.stencil(lambda a, c: 0.3 * (a[1 - c] + a[0] + a[c - 1]))(A, c, out=B)
        return B

    def test_impl3(n):
        A = np.arange(n)
        B = np.zeros(n)
        c = 2
        numba.stencil(lambda a, c: 0.3 * (a[-c + 1] + a[0] + a[c - 1]))(A, c, out=B)
        return B

    def test_impl4(n):
        A = np.arange(n)
        B = np.zeros(n)
        d = 1
        c = 2
        numba.stencil(lambda a, c, d: 0.3 * (a[-c + d] + a[0] + a[c - d]))(A, c, d, out=B)
        return B

    def test_impl_seq(n):
        A = np.arange(n)
        B = np.zeros(n)
        c = 1
        for i in range(1, n - 1):
            B[i] = 0.3 * (A[i - c] + A[i] + A[i + c])
        return B
    n = 100
    cpfunc1 = self.compile_parallel(test_impl1, (types.intp,))
    cpfunc2 = self.compile_parallel(test_impl2, (types.intp,))
    cpfunc3 = self.compile_parallel(test_impl3, (types.intp,))
    cpfunc4 = self.compile_parallel(test_impl4, (types.intp,))
    expected = test_impl_seq(n)
    parfor_output1 = cpfunc1.entry_point(n)
    parfor_output2 = cpfunc2.entry_point(n)
    parfor_output3 = cpfunc3.entry_point(n)
    parfor_output4 = cpfunc4.entry_point(n)
    np.testing.assert_almost_equal(parfor_output1, expected, decimal=3)
    np.testing.assert_almost_equal(parfor_output2, expected, decimal=3)
    np.testing.assert_almost_equal(parfor_output3, expected, decimal=3)
    np.testing.assert_almost_equal(parfor_output4, expected, decimal=3)
    with self.assertRaises(NumbaValueError) as e:
        test_impl4(4)
    self.assertIn("stencil kernel index is not constant, 'neighborhood' option required", str(e.exception))
    with self.assertRaises((LoweringError, NumbaValueError)) as e:
        njit(test_impl4)(4)
    self.assertIn("stencil kernel index is not constant, 'neighborhood' option required", str(e.exception))