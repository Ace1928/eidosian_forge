import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_negative_strided_issue(self):
    h_arr = np.random.random(10)
    c_arr = cuda.to_device(h_arr)

    def base_offset(orig, sliced):
        return sliced['data'][0] - orig['data'][0]
    h_ai = h_arr.__array_interface__
    c_ai = c_arr.__cuda_array_interface__
    h_ai_sliced = h_arr[::-1].__array_interface__
    c_ai_sliced = c_arr[::-1].__cuda_array_interface__
    self.assertEqual(base_offset(h_ai, h_ai_sliced), base_offset(c_ai, c_ai_sliced))
    self.assertEqual(h_ai_sliced['shape'], c_ai_sliced['shape'])
    self.assertEqual(h_ai_sliced['strides'], c_ai_sliced['strides'])