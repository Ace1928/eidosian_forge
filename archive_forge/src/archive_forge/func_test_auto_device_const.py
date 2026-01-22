import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def test_auto_device_const(self):
    d, _ = cuda.devicearray.auto_device(2)
    self.assertTrue(np.all(d.copy_to_host() == np.array(2)))