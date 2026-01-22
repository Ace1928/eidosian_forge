import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_xor_global_2(self):
    rand_const = np.random.randint(500)
    ary = np.random.randint(0, 32, size=32).astype(np.uint32).reshape(4, 8)
    orig = ary.copy()
    cuda_func = cuda.jit('void(uint32[:,:], uint32)')(atomic_xor_global_2)
    cuda_func[1, (4, 8)](ary, rand_const)
    np.testing.assert_equal(ary, orig ^ rand_const)