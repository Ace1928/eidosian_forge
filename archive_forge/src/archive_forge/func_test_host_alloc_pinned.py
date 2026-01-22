import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def test_host_alloc_pinned(self):
    ary = cuda.pinned_array(10, dtype=np.uint32)
    ary.fill(123)
    self.assertTrue(all(ary == 123))
    devary = cuda.to_device(ary)
    driver.device_memset(devary, 0, driver.device_memory_size(devary))
    self.assertTrue(all(ary == 123))
    devary.copy_to_host(ary)
    self.assertTrue(all(ary == 0))