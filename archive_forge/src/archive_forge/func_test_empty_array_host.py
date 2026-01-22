import numpy as np
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from numba.cuda.testing import CUDATestCase
import unittest
def test_empty_array_host(self):
    A = np.arange(0, dtype=np.float64) + 1
    expect = A.sum()
    got = sum_reduce(A)
    self.assertEqual(expect, got)