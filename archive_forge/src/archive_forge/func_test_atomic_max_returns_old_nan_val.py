import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_max_returns_old_nan_val(self):

    @cuda.jit
    def kernel(x):
        x[1] = cuda.atomic.max(x, 0, np.nan)
    self._test_atomic_returns_old(kernel, 10)