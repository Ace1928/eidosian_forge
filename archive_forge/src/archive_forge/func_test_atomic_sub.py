import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_sub(self):
    ary = np.random.randint(0, 32, size=32).astype(np.uint32)
    orig = ary.copy()
    cuda_atomic_sub = cuda.jit('void(uint32[:])')(atomic_sub)
    cuda_atomic_sub[1, 32](ary)
    gold = np.zeros(32, dtype=np.uint32)
    for i in range(orig.size):
        gold[orig[i]] -= 1
    self.assertTrue(np.all(ary == gold))