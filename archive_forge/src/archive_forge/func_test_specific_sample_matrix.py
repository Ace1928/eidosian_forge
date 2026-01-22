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
def test_specific_sample_matrix(self):
    inst = TestLinalgBase('specific_sample_matrix')
    sizes = [(7, 1), (11, 5), (5, 11), (3, 3), (1, 7)]
    for size, dtype, order in product(sizes, inst.dtypes, 'FC'):
        m, n = size
        minmn = min(m, n)
        A = inst.specific_sample_matrix(size, dtype, order)
        self.assertEqual(A.shape, size)
        self.assertEqual(np.linalg.matrix_rank(A), minmn)
        if minmn > 1:
            rank = minmn - 1
            A = inst.specific_sample_matrix(size, dtype, order, rank=rank)
            self.assertEqual(A.shape, size)
            self.assertEqual(np.linalg.matrix_rank(A), rank)
        resolution = 5 * np.finfo(dtype).resolution
        A = inst.specific_sample_matrix(size, dtype, order)
        self.assertEqual(A.shape, size)
        np.testing.assert_allclose(np.linalg.cond(A), 1.0, rtol=resolution, atol=resolution)
        if minmn > 1:
            condition = 10.0
            A = inst.specific_sample_matrix(size, dtype, order, condition=condition)
            self.assertEqual(A.shape, size)
            np.testing.assert_allclose(np.linalg.cond(A), 10.0, rtol=resolution, atol=resolution)

    def check_error(args, msg, err=ValueError):
        with self.assertRaises(err) as raises:
            inst.specific_sample_matrix(*args)
        self.assertIn(msg, str(raises.exception))
    with self.assertRaises(AssertionError) as raises:
        msg = 'blank'
        check_error(((2, 3), np.float64, 'F'), msg, err=ValueError)
    msg = 'size must be a length 2 tuple.'
    check_error(((1,), np.float64, 'F'), msg, err=ValueError)
    msg = "order must be one of 'F' or 'C'."
    check_error(((2, 3), np.float64, 'z'), msg, err=ValueError)
    msg = 'dtype must be a numpy floating point type.'
    check_error(((2, 3), np.int32, 'F'), msg, err=ValueError)
    msg = 'Only one of rank or condition can be specified.'
    check_error(((2, 3), np.float64, 'F', 1, 1), msg, err=ValueError)
    msg = 'Condition number must be >=1.'
    check_error(((2, 3), np.float64, 'F', None, -1), msg, err=ValueError)
    msg = 'Negative dimensions given for matrix shape.'
    check_error(((2, -3), np.float64, 'F'), msg, err=ValueError)
    msg = 'Rank must be greater than zero.'
    check_error(((2, 3), np.float64, 'F', -1), msg, err=ValueError)
    msg = 'Rank given greater than full rank.'
    check_error(((2, 3), np.float64, 'F', 4), msg, err=ValueError)
    msg = 'Condition number was specified for a vector (always 1.).'
    check_error(((1, 3), np.float64, 'F', None, 10), msg, err=ValueError)
    msg = 'Rank must an integer.'
    check_error(((2, 3), np.float64, 'F', 1.5), msg, err=ValueError)