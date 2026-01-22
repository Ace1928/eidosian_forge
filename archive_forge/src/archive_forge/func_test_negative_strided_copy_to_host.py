import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_negative_strided_copy_to_host(self):
    h_arr = np.random.random(10)
    c_arr = cuda.to_device(h_arr)
    sliced = c_arr[::-1]
    with self.assertRaises(NotImplementedError) as raises:
        sliced.copy_to_host()
    expected_msg = 'D->H copy not implemented for negative strides'
    self.assertIn(expected_msg, str(raises.exception))