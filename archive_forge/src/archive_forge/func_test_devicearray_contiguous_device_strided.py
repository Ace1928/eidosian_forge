import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_contiguous_device_strided(self):
    d = cuda.to_device(np.arange(20))
    arr = np.arange(20)
    with self.assertRaises(ValueError) as e:
        d.copy_to_device(cuda.to_device(arr)[::2])
    self.assertEqual(devicearray.errmsg_contiguous_buffer, str(e.exception))