from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
@cuda.jit
def slice_nonunit_stride(x):
    dynsmem = cuda.shared.array(0, dtype=int32)
    sm1 = dynsmem[::2]
    dynsmem[0] = 99
    dynsmem[1] = 99
    dynsmem[2] = 99
    dynsmem[3] = 99
    dynsmem[4] = 99
    dynsmem[5] = 99
    sm1[0] = 1
    sm1[1] = 2
    sm1[2] = 3
    x[0] = dynsmem[0]
    x[1] = dynsmem[1]
    x[2] = dynsmem[2]
    x[3] = dynsmem[3]
    x[4] = dynsmem[4]
    x[5] = dynsmem[5]