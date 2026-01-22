from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def b2d(bs):
    return bs