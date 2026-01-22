import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_add_float_2(self):
    ary = np.random.randint(0, 32, size=32).astype(np.float32).reshape(4, 8)
    ary_wrap = ary.copy()
    orig = ary.copy()
    cuda_atomic_add2 = cuda.jit('void(float32[:,:])')(atomic_add_float_2)
    cuda_atomic_add2[1, (4, 8)](ary)
    cuda_func_wrap = cuda.jit('void(float32[:,:])')(atomic_add_float_2_wrap)
    cuda_func_wrap[1, (4, 8)](ary_wrap)
    self.assertTrue(np.all(ary == orig + 1))
    self.assertTrue(np.all(ary_wrap == orig + 1))