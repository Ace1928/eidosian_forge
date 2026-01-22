import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('cudasim ignores casting by jit decorator signature')
def test_device_casting(self):

    @cuda.jit('int32(int32, int32, int32, int32)', device=True)
    def rgba(r, g, b, a):
        return (r & 255) << 16 | (g & 255) << 8 | (b & 255) << 0 | (a & 255) << 24

    @cuda.jit
    def rgba_caller(x, channels):
        x[0] = rgba(channels[0], channels[1], channels[2], channels[3])
    x = cuda.device_array(1, dtype=np.int32)
    channels = cuda.to_device(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    rgba_caller[1, 1](x, channels)
    self.assertEqual(67174915, x[0])