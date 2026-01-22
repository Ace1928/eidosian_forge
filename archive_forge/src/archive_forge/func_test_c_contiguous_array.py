import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_c_contiguous_array(self):
    ary = np.arange(100)
    arysz = ary.dtype.itemsize * ary.size
    s, e = driver.host_memory_extents(ary)
    self.assertTrue(ary.ctypes.data == s)
    self.assertTrue(arysz == driver.host_memory_size(ary))