import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba import cuda, float64
import unittest
def test_eager_opt(self):
    sig = (float64[::1],)
    kernel = cuda.jit(sig)(kernel_func)
    ptx = kernel.inspect_asm()
    for fragment in removed_by_opt:
        with self.subTest(fragment=fragment):
            self.assertNotIn(fragment, ptx[sig])