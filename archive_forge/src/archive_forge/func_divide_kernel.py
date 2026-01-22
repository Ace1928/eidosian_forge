from numba import cuda, float32, int32
from numba.core.errors import NumbaInvalidConfigWarning
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import ignore_internal_warnings
import re
import unittest
import warnings
@cuda.jit(sig, lineinfo=True)
def divide_kernel(x, y):
    x[0] /= y[0]