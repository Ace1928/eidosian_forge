from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_global_kernel(self):

    def f(r, x, y):
        i = cuda.grid(1)
        if i < len(r):
            r[i] = x[i] + y[i]
    args = (float32[:], float32[:], float32[:])
    ptx, resty = compile_ptx(f, args)
    self.assertNotIn('func_retval', ptx)
    self.assertNotIn('.visible .func', ptx)
    self.assertIn('.visible .entry', ptx)
    self.assertEqual(resty, void)