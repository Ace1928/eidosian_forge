import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_numpy_fill_method(self):

    def test_impl(arr):
        arr.fill(3)
        return arr
    shape = (11, 13)
    arr = np.arange(np.prod(shape)).reshape(shape)
    args = (arr,)
    argtypes = [typeof(x) for x in args]
    sub_pass = self.run_parfor_sub_pass(test_impl, argtypes)
    self.assertEqual(len(sub_pass.rewritten), 1)
    [record] = sub_pass.rewritten
    self.assertEqual(record['reason'], 'loop')
    self.check_records(sub_pass.rewritten)
    self.run_parallel(test_impl, *args)