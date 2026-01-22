import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct
@skip_on_cudasim("Can't check typing in simulator")
def test_numpy_dtype(self):

    @cuda.jit(void(int32[::1]))
    def f(x):
        l = cuda.local.array(10, dtype=np.int32)
        l[0] = x[0]
        x[0] = l[0]
    self.check_dtype(f, int32)