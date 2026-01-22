import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
def switch_gpu():
    with cuda.gpus[1]:
        return cuda.current_context().device.id