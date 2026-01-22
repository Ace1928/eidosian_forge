import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_from_record_like(self):
    hostrec = self.hostz.copy()
    devrec = from_record_like(hostrec)
    self._check_device_record(hostrec, devrec)
    devrec2 = from_record_like(devrec)
    self._check_device_record(devrec, devrec2)
    self.assertNotEqual(devrec.gpu_data, devrec2.gpu_data)