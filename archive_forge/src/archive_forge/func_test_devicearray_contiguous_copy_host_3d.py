import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_contiguous_copy_host_3d(self):
    a_c = np.arange(5 * 5 * 5).reshape(5, 5, 5)
    a_f = np.array(a_c, order='F')
    self._test_devicearray_contiguous_host_copy(a_c, a_f)