from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
@unittest.skip('Needs insert_unresolved_ref support in target')
def test_optional_return(self):
    pfunc = self.mod.make_optional_return_case()
    cfunc = self.mod.make_optional_return_case(cuda.jit)

    @cuda.jit
    def kernel(r, x):
        res = cfunc(x[0])
        if res is None:
            res = 999
        r[0] = res

    def cpu_kernel(x):
        res = pfunc(x)
        if res is None:
            res = 999
        return res
    for arg in (0, 5, 10, 15):
        expected = cpu_kernel(arg)
        x = np.asarray([arg], dtype=np.int64)
        r = np.zeros_like(x)
        kernel[1, 1](r, x)
        actual = r[0]
        self.assertEqual(expected, actual)