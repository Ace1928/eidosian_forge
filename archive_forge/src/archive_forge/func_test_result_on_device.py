import numpy as np
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from numba.cuda.testing import CUDATestCase
import unittest
def test_result_on_device(self):
    A = np.arange(10, dtype=np.float64) + 1
    got = cuda.to_device(np.zeros(1, dtype=np.float64))
    expect = A.sum()
    res = sum_reduce(A, res=got)
    self.assertIsNone(res)
    self.assertEqual(expect, got[0])