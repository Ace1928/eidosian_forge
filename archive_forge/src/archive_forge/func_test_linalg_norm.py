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
def test_linalg_norm(self):
    """
        Test np.linalg.norm
        """
    cfunc = jit(nopython=True)(norm_matrix)

    def check(a, **kwargs):
        expected = norm_matrix(a, **kwargs)
        got = cfunc(a, **kwargs)
        self.assertTrue(not np.iscomplexobj(got))
        resolution = 5 * np.finfo(a.dtype).resolution
        np.testing.assert_allclose(got, expected, rtol=resolution)
        with self.assertNoNRTLeak():
            cfunc(a, **kwargs)
    sizes = [1, 4, 7]
    nrm_types = [None, np.inf, -np.inf, 0, 1, -1, 2, -2, 5, 6.7, -4.3]
    for size, dtype, nrm_type in product(sizes, self.dtypes, nrm_types):
        a = self.sample_vector(size, dtype)
        check(a, ord=nrm_type)
    for dtype, nrm_type in product(self.dtypes, nrm_types):
        a = self.sample_vector(10, dtype)[::3]
        check(a, ord=nrm_type)
    sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
    nrm_types = [None, np.inf, -np.inf, 1, -1, 2, -2]
    for size, dtype, order, nrm_type in product(sizes, self.dtypes, 'FC', nrm_types):
        a = self.specific_sample_matrix(size, dtype, order)
        check(a, ord=nrm_type)
    nrm_types = [None]
    for dtype, nrm_type, order in product(self.dtypes, nrm_types, 'FC'):
        a = self.specific_sample_matrix((17, 13), dtype, order)
        check(a[:3], ord=nrm_type)
        check(a[:, 3:], ord=nrm_type)
        check(a[1, 4::3], ord=nrm_type)
    for dtype, nrm_type, order in product(self.dtypes, nrm_types, 'FC'):
        a = np.empty((0,), dtype=dtype, order=order)
        self.assertEqual(cfunc(a, nrm_type), 0.0)
        a = np.empty((0, 0), dtype=dtype, order=order)
        self.assertEqual(cfunc(a, nrm_type), 0.0)
    rn = 'norm'
    self.assert_wrong_dtype(rn, cfunc, (np.ones((2, 2), dtype=np.int32),))
    self.assert_wrong_dimensions_1D(rn, cfunc, (np.ones(12, dtype=np.float64).reshape(2, 2, 3),))
    self.assert_no_nan_or_inf(cfunc, (np.array([[1.0, 2.0], [np.inf, np.nan]], dtype=np.float64), 2))
    self.assert_invalid_norm_kind(cfunc, (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), 6))