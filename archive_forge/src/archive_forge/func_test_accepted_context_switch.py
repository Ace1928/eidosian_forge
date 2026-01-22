import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
@unittest.skipIf(len(cuda.gpus) < 2, 'need more than 1 gpus')
def test_accepted_context_switch(self):

    def switch_gpu():
        with cuda.gpus[1]:
            return cuda.current_context().device.id
    with cuda.gpus[0]:
        devid = switch_gpu()
    self.assertEqual(int(devid), 1)