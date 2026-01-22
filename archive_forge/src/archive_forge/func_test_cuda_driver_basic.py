from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_cuda_driver_basic(self):
    module = self.context.create_module_ptx(self.ptx)
    function = module.get_function('_Z10helloworldPi')
    array = (c_int * 100)()
    memory = self.context.memalloc(sizeof(array))
    host_to_device(memory, array, sizeof(array))
    ptr = memory.device_ctypes_pointer
    stream = 0
    if _driver.USE_NV_BINDING:
        ptr = c_void_p(int(ptr))
        stream = _driver.binding.CUstream(stream)
    launch_kernel(function.handle, 1, 1, 1, 100, 1, 1, 0, stream, [ptr])
    device_to_host(array, memory, sizeof(array))
    for i, v in enumerate(array):
        self.assertEqual(i, v)
    module.unload()