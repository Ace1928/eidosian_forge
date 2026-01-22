import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice
def test_sad(self):
    x = np.arange(0, 200, 2)
    y = np.arange(50, 150)
    z = np.arange(15, 115)
    r = np.zeros_like(x)
    cufunc = cuda.jit(use_sad)
    cufunc[4, 32](r, x, y, z)
    np.testing.assert_array_equal(np.abs(x - y) + z, r)