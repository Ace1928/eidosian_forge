import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_ctypes_struct(self):

    class mystruct(ctypes.Structure):
        _fields_ = [('x', ctypes.c_int), ('y', ctypes.c_int)]
    data = mystruct(x=123, y=432)
    sz = driver.host_memory_size(data)
    self.assertTrue(ctypes.sizeof(data) == sz)