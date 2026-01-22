import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
from numba.core import config
@xfail_unless_cudasim
def test_raise_in_device_function(self):
    msg = 'Device Function Error'

    @cuda.jit(device=True)
    def f():
        raise ValueError(msg)

    @cuda.jit(debug=True)
    def kernel():
        f()
    with self.assertRaises(ValueError) as raises:
        kernel[1, 1]()
    self.assertIn(msg, str(raises.exception))