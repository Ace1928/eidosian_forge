import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
@needs_lapack
def test_linalg_solve(self):
    """
        Test np.linalg.solve
        """
    cfunc = jit(nopython=True)(solve_system)

    def check(a, b, **kwargs):
        expected = solve_system(a, b, **kwargs)
        got = cfunc(a, b, **kwargs)
        self.assert_contig_sanity(got, 'F')
        use_reconstruction = False
        try:
            np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
        except AssertionError:
            use_reconstruction = True
        if use_reconstruction:
            self.assertEqual(got.shape, expected.shape)
            rec = np.dot(a, got)
            resolution = np.finfo(a.dtype).resolution
            np.testing.assert_allclose(b, rec, rtol=10 * resolution, atol=100 * resolution)
        with self.assertNoNRTLeak():
            cfunc(a, b, **kwargs)
    sizes = [(1, 1), (3, 3), (7, 7)]
    for size, dtype, order in product(sizes, self.dtypes, 'FC'):
        A = self.specific_sample_matrix(size, dtype, order)
        b_sizes = (1, 13)
        for b_size, b_order in product(b_sizes, 'FC'):
            B = self.specific_sample_matrix((A.shape[0], b_size), dtype, b_order)
            check(A, B)
            tmp = B[:, 0].copy(order=b_order)
            check(A, tmp)
    cfunc(np.empty((0, 0)), np.empty((0,)))
    ok = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cfunc(ok, ok)
    rn = 'solve'
    bad = np.array([[1, 0], [0, 1]], dtype=np.int32)
    self.assert_wrong_dtype(rn, cfunc, (ok, bad))
    self.assert_wrong_dtype(rn, cfunc, (bad, ok))
    bad = np.array([[1, 2], [3, 4]], dtype=np.float32)
    self.assert_homogeneous_dtypes(rn, cfunc, (ok, bad))
    self.assert_homogeneous_dtypes(rn, cfunc, (bad, ok))
    bad = np.array([1, 0], dtype=np.float64)
    self.assert_wrong_dimensions(rn, cfunc, (bad, ok))
    bad = np.array([[1.0, 0.0], [np.inf, np.nan]], dtype=np.float64)
    self.assert_no_nan_or_inf(cfunc, (ok, bad))
    self.assert_no_nan_or_inf(cfunc, (bad, ok))
    ok_oneD = np.array([1.0, 2.0], dtype=np.float64)
    cfunc(ok, ok_oneD)
    bad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    self.assert_wrong_dimensions_1D(rn, cfunc, (ok, bad))
    bad1D = np.array([1.0], dtype=np.float64)
    bad2D = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    self.assert_dimensionally_invalid(cfunc, (ok, bad1D))
    self.assert_dimensionally_invalid(cfunc, (ok, bad2D))
    bad2D = self.specific_sample_matrix((2, 2), np.float64, 'C', rank=1)
    self.assert_raise_on_singular(cfunc, (bad2D, ok))