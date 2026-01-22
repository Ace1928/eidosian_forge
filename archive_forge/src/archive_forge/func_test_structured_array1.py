import numpy as np
import ctypes
from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import cuda
def test_structured_array1(self):
    ary = self.sample1d
    for i in range(self.sample1d.size):
        x = i + 1
        self.assertEqual(ary[i]['a'], x / 2)
        self.assertEqual(ary[i]['b'], x)
        self.assertEqual(ary[i]['c'], x * 1j)
        self.assertEqual(ary[i]['d'], str(x) * N_CHARS)