from math import sqrt
from numba import cuda, float32, int16, int32, int64, uint32, void
from numba.cuda import compile_ptx, compile_ptx_for_current_device
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_nanosleep(self):

    def use_nanosleep(x):
        cuda.nanosleep(32)
        cuda.nanosleep(x)
    ptx, resty = compile_ptx(use_nanosleep, (uint32,), cc=(7, 0))
    nanosleep_count = 0
    for line in ptx.split('\n'):
        if 'nanosleep.u32' in line:
            nanosleep_count += 1
    expected = 2
    self.assertEqual(expected, nanosleep_count, f'Got {nanosleep_count} nanosleep instructions, expected {expected}')