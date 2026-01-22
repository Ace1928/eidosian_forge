from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_c_abi_disallowed_for_kernel(self):

    def f(x, y):
        return x + y
    with self.assertRaisesRegex(NotImplementedError, 'The C ABI is not supported for kernels'):
        compile_ptx(f, (int32, int32), abi='c')