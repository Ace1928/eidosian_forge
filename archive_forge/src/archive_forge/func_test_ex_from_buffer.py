import unittest
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim)
from numba.tests.support import skip_unless_cffi
def test_ex_from_buffer(self):
    from numba import cuda
    import os
    basedir = os.path.dirname(os.path.abspath(__file__))
    functions_cu = os.path.join(basedir, 'ffi', 'functions.cu')
    signature = 'float32(CPointer(float32), int32)'
    sum_reduce = cuda.declare_device('sum_reduce', signature)
    import cffi
    ffi = cffi.FFI()

    @cuda.jit(link=[functions_cu])
    def reduction_caller(result, array):
        array_ptr = ffi.from_buffer(array)
        result[()] = sum_reduce(array_ptr, len(array))
    import numpy as np
    x = np.arange(10).astype(np.float32)
    r = np.ndarray((), dtype=np.float32)
    reduction_caller[1, 1](r, x)
    expected = np.sum(x)
    actual = r[()]
    np.testing.assert_allclose(expected, actual)