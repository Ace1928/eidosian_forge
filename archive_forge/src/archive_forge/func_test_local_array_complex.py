import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct
def test_local_array_complex(self):
    sig = 'void(complex128[:], complex128[:])'
    jculocalcomplex = cuda.jit(sig)(culocalcomplex)
    A = (np.arange(100, dtype='complex128') - 1) / 2j
    B = np.zeros_like(A)
    jculocalcomplex[1, 1](A, B)
    self.assertTrue(np.all(A == B))