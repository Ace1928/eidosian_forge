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
@needs_blas
def test_contiguity_warnings(self):
    m, k, n = (2, 3, 4)
    dtype = np.float64
    a = self.sample_matrix(m, k, dtype)[::-1]
    b = self.sample_matrix(k, n, dtype)[::-1]
    out = np.empty((m, n), dtype)
    cfunc = jit(nopython=True)(dot2)
    with self.check_contiguity_warning(cfunc.py_func):
        cfunc(a, b)
    cfunc = jit(nopython=True)(dot3)
    with self.check_contiguity_warning(cfunc.py_func):
        cfunc(a, b, out)
    a = self.sample_vector(n, dtype)[::-1]
    b = self.sample_vector(n, dtype)[::-1]
    cfunc = jit(nopython=True)(vdot)
    with self.check_contiguity_warning(cfunc.py_func):
        cfunc(a, b)