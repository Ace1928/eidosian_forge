import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_setitem_full_slice(self):

    def test_impl():
        n = 10
        a = np.ones(n)
        a[:] = 7
        return a
    sub_pass = self.run_parfor_sub_pass(test_impl, ())
    self.assertEqual(len(sub_pass.rewritten), 1)
    [record] = sub_pass.rewritten
    self.assertEqual(record['reason'], 'slice')
    self.check_records(sub_pass.rewritten)
    self.run_parallel(test_impl)