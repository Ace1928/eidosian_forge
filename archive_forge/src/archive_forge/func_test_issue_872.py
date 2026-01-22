import numpy as np
from numba import cuda, float64
from numba.cuda.testing import unittest, CUDATestCase
def test_issue_872(self):
    """
        Ensure that typing and lowering of CUDA kernel API primitives works in
        more than one block. Was originally to ensure that macro expansion works
        for more than one block (issue #872), but macro expansion has been
        replaced by a "proper" implementation of all kernel API functions.
        """

    @cuda.jit('void(float64[:,:])')
    def cuda_kernel_api_in_multiple_blocks(ary):
        for i in range(2):
            tx = cuda.threadIdx.x
        for j in range(3):
            ty = cuda.threadIdx.y
        sm = cuda.shared.array((2, 3), float64)
        sm[tx, ty] = 1.0
        ary[tx, ty] = sm[tx, ty]
    a = np.zeros((2, 3))
    cuda_kernel_api_in_multiple_blocks[1, (2, 3)](a)