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
def check_slogdet(self, cfunc, a, **kwargs):
    expected = slogdet_matrix(a, **kwargs)
    got = cfunc(a, **kwargs)
    self.assertEqual(len(expected), len(got))
    self.assertEqual(len(got), 2)
    for k in range(2):
        self.assertEqual(np.iscomplexobj(got[k]), np.iscomplexobj(expected[k]))
    got_conv = a.dtype.type(got[0])
    np.testing.assert_array_almost_equal_nulp(got_conv, expected[0], nulp=10)
    resolution = 5 * np.finfo(a.dtype).resolution
    np.testing.assert_allclose(got[1], expected[1], rtol=resolution, atol=resolution)
    with self.assertNoNRTLeak():
        cfunc(a, **kwargs)