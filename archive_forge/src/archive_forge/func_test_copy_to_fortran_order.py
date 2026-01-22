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
def test_copy_to_fortran_order(self):
    from numba.np.linalg import _copy_to_fortran_order

    def check(udt, expectfn, shapes, dtypes, orders):
        for shape, dtype, order in product(shapes, dtypes, orders):
            a = np.arange(np.prod(shape)).reshape(shape, order=order)
            r = udt(a)
            self.assertPreciseEqual(expectfn(a), r)
            self.assertNotEqual(a.ctypes.data, r.ctypes.data)

    @njit
    def direct_call(a):
        return _copy_to_fortran_order(a)
    shapes = [(3, 4), (3, 2, 5)]
    dtypes = [np.intp]
    orders = ['C', 'F']
    check(direct_call, np.asfortranarray, shapes, dtypes, orders)

    @njit
    def slice_to_any(a):
        sliced = a[::2][0]
        return _copy_to_fortran_order(sliced)
    shapes = [(3, 3, 4), (3, 3, 2, 5)]
    dtypes = [np.intp]
    orders = ['C', 'F']

    def expected_slice_to_any(a):
        sliced = a[::2][0]
        return np.asfortranarray(sliced)
    check(slice_to_any, expected_slice_to_any, shapes, dtypes, orders)