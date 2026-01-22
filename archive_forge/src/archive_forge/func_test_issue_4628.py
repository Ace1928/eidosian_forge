import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
@skip_on_cudasim('Kernel overloads not created in the simulator')
def test_issue_4628(self):

    @cuda.jit
    def func(A, out):
        i = cuda.grid(1)
        out[i] = A[i] * 2
    n = 128
    a = np.ones((n,))
    d_a = cuda.to_device(a)
    result = np.zeros((n,))
    func[1, 128](a, result)
    func[1, 128](d_a, result)
    self.assertEqual(1, len(func.overloads))