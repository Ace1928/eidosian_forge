from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
def test_no_lineinfo_in_device_function(self):

    @cuda.jit
    def callee(x):
        x[0] += 1

    @cuda.jit
    def caller(x):
        x[0] = 1
        callee(x)
    sig = (int32[:],)
    self._check(caller, sig=sig, expect=False)