import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
def test_record_access(self):
    backyard_type = [('statue', np.float64), ('newspaper', np.float64, (6,))]
    goose_type = [('garden', np.float64, (12,)), ('town', np.float64, (42,)), ('backyard', backyard_type)]
    goose_np_type = np.dtype(goose_type, align=True)

    @cuda.jit
    def simple_kernel(f):
        f.garden[0] = 45.0
        f.backyard.newspaper[3] = 2.0
        f.backyard.newspaper[3] = f.backyard.newspaper[3] + 3.0
    item = np.recarray(1, dtype=goose_np_type)
    simple_kernel[1, 1](item[0])
    np.testing.assert_equal(item[0]['garden'][0], 45)
    np.testing.assert_equal(item[0]['backyard']['newspaper'][3], 5)