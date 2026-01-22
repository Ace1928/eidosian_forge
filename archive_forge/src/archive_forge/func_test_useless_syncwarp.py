import numpy as np
from numba import cuda, int32, float32
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba.core.config import ENABLE_CUDASIM
@skip_on_cudasim('syncwarp not implemented on cudasim')
def test_useless_syncwarp(self):
    self._test_useless(useless_syncwarp)