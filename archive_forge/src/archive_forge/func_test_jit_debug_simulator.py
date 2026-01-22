from numba import cuda
import numpy as np
from numba.cuda.testing import CUDATestCase
from numba.tests.support import override_config
import unittest
def test_jit_debug_simulator(self):
    with override_config('ENABLE_CUDASIM', 1):

        @cuda.jit(debug=True)
        def f(x):
            pass