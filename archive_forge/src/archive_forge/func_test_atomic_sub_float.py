import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_sub_float(self):
    ary = np.random.randint(0, 32, size=32).astype(np.float32)
    orig = ary.copy().astype(np.intp)
    cuda_atomic_sub_float = cuda.jit('void(float32[:])')(atomic_sub_float)
    cuda_atomic_sub_float[1, 32](ary)
    gold = np.zeros(32, dtype=np.float32)
    for i in range(orig.size):
        gold[orig[i]] -= 1.0
    self.assertTrue(np.all(ary == gold))