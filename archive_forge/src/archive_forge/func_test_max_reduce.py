import numpy as np
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from numba.cuda.testing import CUDATestCase
import unittest
def test_max_reduce(self):
    max_reduce = cuda.Reduce(lambda a, b: max(a, b))
    A = np.arange(3717, dtype=np.float64) + 1
    expect = A.max()
    got = max_reduce(A, init=0)
    self.assertEqual(expect, got)