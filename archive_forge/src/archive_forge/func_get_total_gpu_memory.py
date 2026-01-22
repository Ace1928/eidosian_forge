import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only
def get_total_gpu_memory(self):
    if USE_NV_BINDING:
        free, total = driver.cuMemGetInfo()
        return total
    else:
        free = c_size_t()
        total = c_size_t()
        driver.cuMemGetInfo(byref(free), byref(total))
        return total.value