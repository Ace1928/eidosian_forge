import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def test_host_alloc_driver(self):
    n = 32
    mem = cuda.current_context().memhostalloc(n, mapped=True)
    dtype = np.dtype(np.uint8)
    ary = np.ndarray(shape=n // dtype.itemsize, dtype=dtype, buffer=mem)
    magic = 171
    driver.device_memset(mem, magic, n)
    self.assertTrue(np.all(ary == magic))
    ary.fill(n)
    recv = np.empty_like(ary)
    driver.device_to_host(recv, mem, ary.size)
    self.assertTrue(np.all(ary == recv))
    self.assertTrue(np.all(recv == n))