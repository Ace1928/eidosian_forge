import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_add_double_global_2(self):
    ary = np.random.randint(0, 32, size=32).astype(np.float64).reshape(4, 8)
    ary_wrap = ary.copy()
    orig = ary.copy()
    sig = 'void(float64[:,:])'
    cuda_func = cuda.jit(sig)(atomic_add_double_global_2)
    wrap_cuda_func = cuda.jit(sig)(atomic_add_double_global_2_wrap)
    cuda_func[1, (4, 8)](ary)
    wrap_cuda_func[1, (4, 8)](ary_wrap)
    np.testing.assert_equal(ary, orig + 1)
    np.testing.assert_equal(ary_wrap, orig + 1)
    self.assertCorrectFloat64Atomics(cuda_func, shared=False)
    self.assertCorrectFloat64Atomics(wrap_cuda_func, shared=False)