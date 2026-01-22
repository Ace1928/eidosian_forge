from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_issue_2393(self):
    """
        Test issue of warp misalign address due to nvvm not knowing the
        alignment(? but it should have taken the natural alignment of the type)
        """
    num_weights = 2
    num_blocks = 48
    examples_per_block = 4
    threads_per_block = 1

    @cuda.jit
    def costs_func(d_block_costs):
        s_features = cuda.shared.array((examples_per_block, num_weights), float64)
        s_initialcost = cuda.shared.array(7, float64)
        threadIdx = cuda.threadIdx.x
        prediction = 0
        for j in range(num_weights):
            prediction += s_features[threadIdx, j]
        d_block_costs[0] = s_initialcost[0] + prediction
    block_costs = np.zeros(num_blocks, dtype=np.float64)
    d_block_costs = cuda.to_device(block_costs)
    costs_func[num_blocks, threads_per_block](d_block_costs)
    cuda.synchronize()