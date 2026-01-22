from __future__ import print_function, absolute_import, division
import unittest
import numpy as np
from numba import guvectorize
from numba.tests.support import TestCase
def test_update_inplace_parallel(self):
    gufunc = guvectorize(['void(f8[:], f8[:])'], '(t),()', nopython=True, writable_args=(0,), target='parallel')(py_replace_2nd)
    self._run_test_for_gufunc(gufunc, py_replace_2nd)