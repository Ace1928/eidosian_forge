import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def check_atomic_nanmax(self, dtype, lo, hi, init_val):
    vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
    vals[1::2] = init_val
    res = np.zeros(1, dtype=vals.dtype)
    cuda_func = cuda.jit(atomic_nanmax)
    cuda_func[32, 32](res, vals)
    gold = np.nanmax(vals)
    np.testing.assert_equal(res, gold)