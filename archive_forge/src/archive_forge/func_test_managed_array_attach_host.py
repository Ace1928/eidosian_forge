import numpy as np
from ctypes import byref, c_size_t
from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim, skip_on_arm
from numba.tests.support import linux_only
def test_managed_array_attach_host(self):
    self._test_managed_array()
    msg = 'Host attached managed memory is not accessible prior to CC 6.0'
    self.skip_if_cc_major_lt(6, msg)
    self._test_managed_array(attach_global=False)