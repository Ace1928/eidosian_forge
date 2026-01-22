import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def test_host_operators(self):
    for ary in [cuda.mapped_array(10, dtype=np.uint32), cuda.pinned_array(10, dtype=np.uint32)]:
        ary[:] = range(10)
        self.assertTrue(sum(ary + 1) == 55)
        self.assertTrue(sum((ary + 1) * 2 - 1) == 100)
        self.assertTrue(sum(ary < 5) == 5)
        self.assertTrue(sum(ary <= 5) == 6)
        self.assertTrue(sum(ary > 6) == 3)
        self.assertTrue(sum(ary >= 6) == 4)
        self.assertTrue(sum(ary ** 2) == 285)
        self.assertTrue(sum(ary // 2) == 20)
        self.assertTrue(sum(ary / 2.0) == 22.5)
        self.assertTrue(sum(ary % 2) == 5)