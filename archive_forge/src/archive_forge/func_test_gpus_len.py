import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
def test_gpus_len(self):
    self.assertGreater(len(cuda.gpus), 0)