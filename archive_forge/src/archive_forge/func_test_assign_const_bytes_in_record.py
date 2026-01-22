import re
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from llvmlite import ir
def test_assign_const_bytes_in_record(self):

    @cuda.jit
    def f(a):
        a[0]['x'] = 1
        a[0]['y'] = b'ABC'
        a[1]['x'] = 2
        a[1]['y'] = b'XYZ'
    dt = np.dtype([('x', np.float32), ('y', np.dtype('S12'))])
    a = np.zeros(2, dt)
    f[1, 1](a)
    reference = np.asarray([(1, b'ABC'), (2, b'XYZ')], dtype=dt)
    np.testing.assert_array_equal(reference, a)