import numpy as np
from numba import cuda
from numba.cuda.args import wrap_arg
from numba.cuda.testing import CUDATestCase
import unittest
def test_array_default(self):
    host_arr = np.zeros(1, dtype=np.int64)
    self.set_array_to_three[1, 1](host_arr)
    self.assertEqual(3, host_arr[0])