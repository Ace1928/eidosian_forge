import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def test_set_a(self):
    self._test_set_equal(set_a, 3.1415, types.float64)
    self._test_set_equal(set_a, 3.0, types.float32)