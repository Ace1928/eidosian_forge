import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_nanmax_double_shared(self):
    vals = np.random.randint(0, 32, size=32).astype(np.float64)
    vals[1::2] = np.nan
    res = np.array([0], dtype=vals.dtype)
    sig = 'void(float64[:], float64[:])'
    cuda_func = cuda.jit(sig)(atomic_nanmax_double_shared)
    cuda_func[1, 32](res, vals)
    gold = np.nanmax(vals)
    np.testing.assert_equal(res, gold)