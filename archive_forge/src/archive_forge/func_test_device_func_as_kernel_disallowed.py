import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('cudasim will allow calling any function')
def test_device_func_as_kernel_disallowed(self):

    @cuda.jit(device=True)
    def f():
        pass
    with self.assertRaises(RuntimeError) as raises:
        f[1, 1]()
    self.assertIn('Cannot compile a device function as a kernel', str(raises.exception))