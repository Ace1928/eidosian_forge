import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_consume_stream(self):
    s = cuda.stream()
    f_arr = ForeignArray(cuda.device_array(10, stream=s))
    c_arr = cuda.as_cuda_array(f_arr)
    self.assertTrue(c_arr.stream.external)
    stream_value = self.get_stream_value(s)
    imported_stream_value = self.get_stream_value(c_arr.stream)
    self.assertEqual(stream_value, imported_stream_value)