from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_shared_recarray(self):
    arr = np.recarray(128, dtype=recordwith2darray)
    for x in range(len(arr)):
        arr[x].i = x
        j = np.arange(3 * 2, dtype=np.float32)
        arr[x].j = j.reshape(3, 2) * x
    self._test_shared(arr)