import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
@skip_unless_cudasim('Only works on CUDASIM')
def test_deadlock_on_exception(self):

    def assert_no_blockthreads():
        blockthreads = []
        for t in threading.enumerate():
            if not isinstance(t, simulator.kernel.BlockThread):
                continue
            t.join(1)
            if t.is_alive():
                self.fail('Blocked kernel thread: %s' % t)
        self.assertListEqual(blockthreads, [])

    @simulator.jit
    def assign_with_sync(x, y):
        i = cuda.grid(1)
        y[i] = x[i]
        cuda.syncthreads()
        cuda.syncthreads()
    x = np.arange(3)
    y = np.empty(3)
    assign_with_sync[1, 3](x, y)
    np.testing.assert_array_equal(x, y)
    assert_no_blockthreads()
    with self.assertRaises(IndexError):
        assign_with_sync[1, 6](x, y)
    assert_no_blockthreads()