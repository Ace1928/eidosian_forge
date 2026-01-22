import math
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.random import \
def test_uniform_float32(self):
    self.check_uniform(rng_kernel_float32, np.float32)