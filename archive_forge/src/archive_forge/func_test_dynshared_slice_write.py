from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_dynshared_slice_write(self):

    @cuda.jit
    def slice_write(x):
        dynsmem = cuda.shared.array(0, dtype=int32)
        sm1 = dynsmem[0:1]
        sm2 = dynsmem[1:2]
        sm1[0] = 1
        sm2[0] = 2
        x[0] = dynsmem[0]
        x[1] = dynsmem[1]
    arr = np.zeros(2, dtype=np.int32)
    expected = np.array([1, 2], dtype=np.int32)
    self._test_dynshared_slice(slice_write, arr, expected)