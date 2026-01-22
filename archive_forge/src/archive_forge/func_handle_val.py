import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def handle_val(mem):
    if driver.USE_NV_BINDING:
        return int(mem.handle)
    else:
        return mem.handle.value