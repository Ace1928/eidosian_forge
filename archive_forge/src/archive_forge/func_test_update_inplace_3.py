from __future__ import print_function, absolute_import, division
import unittest
import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase
def test_update_inplace_3(self):
    gufunc = guvectorize(['void(f8[:], f8[:], f8[:], f8[:])'], '(t),(t),(t),()', nopython=True)(py_update_3)
    self._run_test_for_gufunc(gufunc, py_update_3, expect_f4_to_pass=False)
    gufunc = guvectorize(['void(f8[:], f8[:], f8[:], f8[:])'], '(t),(t),(t),()', nopython=True, writable_args=(0, 1, 2))(py_update_3)
    self._run_test_for_gufunc(gufunc, py_update_3)
    gufunc = guvectorize(['void(f8[:], f8[:], f8[:], f8[:])'], '(t),(t),(t),()', nopython=True, writable_args=('x0_t', 'x1_t', 2))(py_update_3)
    self._run_test_for_gufunc(gufunc, py_update_3)