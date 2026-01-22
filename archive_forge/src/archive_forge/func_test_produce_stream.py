import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_produce_stream(self):
    s = cuda.stream()
    c_arr = cuda.device_array(10, stream=s)
    cai_stream = c_arr.__cuda_array_interface__['stream']
    stream_value = self.get_stream_value(s)
    self.assertEqual(stream_value, cai_stream)
    s = cuda.stream()
    mapped_arr = cuda.mapped_array(10, stream=s)
    cai_stream = mapped_arr.__cuda_array_interface__['stream']
    stream_value = self.get_stream_value(s)
    self.assertEqual(stream_value, cai_stream)