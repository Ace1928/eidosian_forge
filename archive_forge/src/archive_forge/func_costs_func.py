from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
@cuda.jit
def costs_func(d_block_costs):
    s_features = cuda.shared.array((examples_per_block, num_weights), float64)
    s_initialcost = cuda.shared.array(7, float64)
    threadIdx = cuda.threadIdx.x
    prediction = 0
    for j in range(num_weights):
        prediction += s_features[threadIdx, j]
    d_block_costs[0] = s_initialcost[0] + prediction