import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_bug6697(self):
    ary = np.arange(10, dtype=np.int16)
    dary = cuda.to_device(ary)
    got = np.asarray(dary)
    self.assertEqual(got.dtype, dary.dtype)