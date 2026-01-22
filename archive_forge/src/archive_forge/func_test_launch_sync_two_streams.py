import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_launch_sync_two_streams(self):
    s1 = cuda.stream()
    s2 = cuda.stream()
    f_arr1 = ForeignArray(cuda.device_array(10, stream=s1))
    f_arr2 = ForeignArray(cuda.device_array(10, stream=s2))

    @cuda.jit
    def f(x, y):
        pass
    with patch.object(cuda.cudadrv.driver.Stream, 'synchronize', return_value=None) as mock_sync:
        f[1, 1](f_arr1, f_arr2)
    mock_sync.assert_has_calls([call(), call()])