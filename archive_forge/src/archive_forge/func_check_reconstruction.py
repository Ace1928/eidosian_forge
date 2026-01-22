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
def check_reconstruction(self, a, got, expected):
    u, sv, vt = got
    for k in range(len(expected)):
        self.assertEqual(got[k].shape, expected[k].shape)
    s = np.zeros((u.shape[1], vt.shape[0]))
    np.fill_diagonal(s, sv)
    rec = np.dot(np.dot(u, s), vt)
    resolution = np.finfo(a.dtype).resolution
    np.testing.assert_allclose(a, rec, rtol=10 * resolution, atol=100 * resolution)