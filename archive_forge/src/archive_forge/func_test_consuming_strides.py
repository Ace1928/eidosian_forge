import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_consuming_strides(self):
    hostarray = np.arange(10).reshape(2, 5)
    devarray = cuda.to_device(hostarray)
    face = devarray.__cuda_array_interface__
    self.assertIsNone(face['strides'])
    got = cuda.from_cuda_array_interface(face).copy_to_host()
    np.testing.assert_array_equal(got, hostarray)
    self.assertTrue(got.flags['C_CONTIGUOUS'])
    face['strides'] = hostarray.strides
    self.assertIsNotNone(face['strides'])
    got = cuda.from_cuda_array_interface(face).copy_to_host()
    np.testing.assert_array_equal(got, hostarray)
    self.assertTrue(got.flags['C_CONTIGUOUS'])