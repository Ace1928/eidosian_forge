import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
@skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
def test_ravel_stride_c(self):
    ary = np.arange(60)
    reshaped = ary.reshape(2, 5, 2, 3)
    dary = cuda.to_device(reshaped)
    darystride = dary[::2, ::2, ::2, ::2]
    dary_data = dary.__cuda_array_interface__['data'][0]
    ddarystride_data = darystride.__cuda_array_interface__['data'][0]
    self.assertEqual(dary_data, ddarystride_data)
    with self.assertRaises(NotImplementedError):
        darystride.ravel()