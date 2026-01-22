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
def assert_is_identity_matrix(self, got, rtol=None, atol=None):
    """
        Checks if a matrix is equal to the identity matrix.
        """
    self.assertEqual(got.shape[-1], got.shape[-2])
    eye = np.eye(got.shape[-1], dtype=got.dtype)
    resolution = 5 * np.finfo(got.dtype).resolution
    if rtol is None:
        rtol = 10 * resolution
    if atol is None:
        atol = 100 * resolution
    np.testing.assert_allclose(got, eye, rtol, atol)