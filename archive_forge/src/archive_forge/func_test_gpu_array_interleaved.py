import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def test_gpu_array_interleaved(self):

    @cuda.jit('void(double[:], double[:])')
    def copykernel(x, y):
        i = cuda.grid(1)
        if i < x.shape[0]:
            x[i] = i
            y[i] = i
    x = np.arange(10, dtype=np.double)
    y = x[:-1:2]
    try:
        cuda.devicearray.auto_device(y)
    except ValueError:
        pass
    else:
        raise AssertionError('Should raise exception complaining the contiguous-ness of the array.')