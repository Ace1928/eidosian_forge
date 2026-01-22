from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_cuda_driver_occupancy(self):
    module = self.context.create_module_ptx(self.ptx)
    function = module.get_function('_Z10helloworldPi')
    value = self.context.get_active_blocks_per_multiprocessor(function, 128, 128)
    self.assertTrue(value > 0)

    def b2d(bs):
        return bs
    grid, block = self.context.get_max_potential_block_size(function, b2d, 128, 128)
    self.assertTrue(grid > 0)
    self.assertTrue(block > 0)