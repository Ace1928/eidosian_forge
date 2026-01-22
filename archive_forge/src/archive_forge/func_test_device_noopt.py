import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba import cuda, float64
import unittest
def test_device_noopt(self):
    sig = (float64, float64, float64)
    device = cuda.jit(sig, device=True, opt=False)(device_func)
    ptx = device.inspect_asm(sig)
    self.assertNotIn('fma.rn.f64', ptx)