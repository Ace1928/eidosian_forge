import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice
def test_frexp(self):
    arr = np.linspace(start=1.0, stop=10.0, num=100, dtype=np.float64)
    fracres = np.zeros_like(arr)
    expres = np.zeros(shape=arr.shape, dtype=np.int32)
    cufunc = cuda.jit(use_frexp)
    cufunc[4, 32](fracres, expres, arr)
    frac_expect, exp_expect = np.frexp(arr)
    np.testing.assert_array_equal(frac_expect, fracres)
    np.testing.assert_array_equal(exp_expect, expres)