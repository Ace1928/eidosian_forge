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
def test_outer(self):
    cfunc = jit(nopython=True)(outer_matrix)

    def check(a, b, **kwargs):
        expected = outer_matrix(a, b)
        got = cfunc(a, b)
        res = 5 * np.finfo(np.asarray(a).dtype).resolution
        np.testing.assert_allclose(got, expected, rtol=res, atol=res)
        if 'out' in kwargs:
            got = cfunc(a, b, **kwargs)
            np.testing.assert_allclose(got, expected, rtol=res, atol=res)
            np.testing.assert_allclose(kwargs['out'], expected, rtol=res, atol=res)
        with self.assertNoNRTLeak():
            cfunc(a, b, **kwargs)
    dts = cycle(self.dtypes)
    for size1, size2 in product(self.sizes, self.sizes):
        dtype = next(dts)
        a, b = self._get_input(size1, size2, dtype)
        check(a, b)
        c = np.empty((np.asarray(a).size, np.asarray(b).size), dtype=np.asarray(a).dtype)
        check(a, b, out=c)
    self._assert_wrong_dim('outer', cfunc)