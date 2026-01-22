from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def three_device_fns(kernel_debug, leaf_debug):

    @cuda.jit(device=True, debug=leaf_debug, opt=False)
    def f3(x):
        return x * x

    @cuda.jit(device=True)
    def f2(x):
        return f3(x) + 1

    @cuda.jit(device=True)
    def f1(x, y):
        return x - f2(y)

    @cuda.jit(debug=kernel_debug, opt=False)
    def kernel(x, y):
        f1(x, y)
    kernel[1, 1](1, 2)