from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
@cuda.jit
def use_sm_chunk_copy(x, y):
    sm = cuda.shared.array(nthreads, dtype=dt)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bd = cuda.blockDim.x
    i = bx * bd + tx
    if i < len(x):
        sm[tx] = x[i]
    cuda.syncthreads()
    if tx == 0:
        for j in range(nthreads):
            y[bd * bx + j] = sm[j]