import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_structured_array4(self):
    arr = np.zeros(1, dtype=recwithrecwithmat)
    d_arr = cuda.to_device(arr)
    d_arr[0]['y']['i'] = 1
    self.assertEqual(d_arr[0]['y']['i'], 1)
    d_arr[0]['y']['j'][0, 0] = 2.0
    self.assertEqual(d_arr[0]['y']['j'][0, 0], 2.0)