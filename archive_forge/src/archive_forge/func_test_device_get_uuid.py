from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_device_get_uuid(self):
    h = '[0-9a-f]{%d}'
    h4 = h % 4
    h8 = h % 8
    h12 = h % 12
    uuid_format = f'^GPU-{h8}-{h4}-{h4}-{h4}-{h12}$'
    dev = devices.get_context().device
    self.assertRegex(dev.uuid, uuid_format)