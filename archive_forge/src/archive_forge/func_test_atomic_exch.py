import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_exch(self):
    rand_const = np.random.randint(50, 100, dtype=np.uint32)
    ary = np.random.randint(0, 32, size=32).astype(np.uint32)
    idx = np.arange(32, dtype=np.uint32)
    cuda_func = cuda.jit('void(uint32[:], uint32[:], uint32)')(atomic_exch)
    cuda_func[1, 32](ary, idx, rand_const)
    np.testing.assert_equal(ary, rand_const)