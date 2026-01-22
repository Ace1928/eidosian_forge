import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only
def test_managed_alloc_driver_undersubscribe(self):
    msg = 'Managed memory unsupported prior to CC 3.0'
    self.skip_if_cc_major_lt(3, msg)
    self._test_managed_alloc_driver(0.5)