import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def test_usecase1(self):
    pyfunc = usecase1
    mystruct_dt = np.dtype([('p', np.float64), ('row', np.float64), ('col', np.float64)])
    mystruct = numpy_support.from_dtype(mystruct_dt)
    cfunc = njit((mystruct[:], mystruct[:]))(pyfunc)
    st1 = np.recarray(3, dtype=mystruct_dt)
    st2 = np.recarray(3, dtype=mystruct_dt)
    st1.p = np.arange(st1.size) + 1
    st1.row = np.arange(st1.size) + 1
    st1.col = np.arange(st1.size) + 1
    st2.p = np.arange(st2.size) + 1
    st2.row = np.arange(st2.size) + 1
    st2.col = np.arange(st2.size) + 1
    expect1 = st1.copy()
    expect2 = st2.copy()
    got1 = expect1.copy()
    got2 = expect2.copy()
    pyfunc(expect1, expect2)
    cfunc(got1, got2)
    np.testing.assert_equal(expect1, got1)
    np.testing.assert_equal(expect2, got2)