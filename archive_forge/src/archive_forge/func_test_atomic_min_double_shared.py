import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_min_double_shared(self):
    vals = np.random.randint(0, 32, size=32).astype(np.float64)
    res = np.ones(1, np.float64) * 32
    sig = 'void(float64[:], float64[:])'
    cuda_func = cuda.jit(sig)(atomic_min_double_shared)
    cuda_func[1, 32](res, vals)
    gold = np.min(vals)
    np.testing.assert_equal(res, gold)