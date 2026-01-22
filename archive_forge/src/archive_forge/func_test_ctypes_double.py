import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_ctypes_double(self):
    data = ctypes.c_double(1.234)
    sz = driver.host_memory_size(data)
    self.assertTrue(ctypes.sizeof(data) == sz)