import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_prange_map_none_index(self):

    def test_impl():
        n = 20
        arr = np.ones(n)
        for i in prange(n):
            inner = arr[i:i + 1]
            inner[()] += 1
        return arr
    sub_pass = self.run_parfor_sub_pass(test_impl, ())
    self.assertEqual(len(sub_pass.rewritten), 1)
    self.check_records(sub_pass.rewritten)
    [record] = sub_pass.rewritten
    self.assertEqual(record['reason'], 'loop')
    self.run_parallel(test_impl)