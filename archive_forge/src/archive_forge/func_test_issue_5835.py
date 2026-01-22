from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def test_issue_5835(self):

    @cuda.jit((types.int32[::1],), debug=True, opt=False)
    def f(x):
        x[0] = 0