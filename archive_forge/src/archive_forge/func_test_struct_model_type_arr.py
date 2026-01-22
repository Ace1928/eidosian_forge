import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct
def test_struct_model_type_arr(self):

    @cuda.jit(void(int32[::1], int32[::1]))
    def f(outx, outy):
        arr = cuda.local.array(10, dtype=test_struct_model_type)
        for i in range(len(arr)):
            obj = TestStruct(int32(i), int32(i * 2))
            arr[i] = obj
        for i in range(len(arr)):
            outx[i] = arr[i].x
            outy[i] = arr[i].y
    arrx = np.array((10,), dtype='int32')
    arry = np.array((10,), dtype='int32')
    f[1, 1](arrx, arry)
    for i, x in enumerate(arrx):
        self.assertEqual(x, i)
    for i, y in enumerate(arry):
        self.assertEqual(y, i * 2)