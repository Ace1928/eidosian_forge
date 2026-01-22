import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def test_rect(self):

    def do_test(tp, seed_values):
        values = [(z.real, z.imag) for z in seed_values if not math.isinf(z.imag) or z.real == 0]
        float_type = tp.underlying_float
        self.run_binary(rect_usecase, [tp(float_type, float_type)], values)
    do_test(types.complex128, self.more_values())
    do_test(types.complex64, self.basic_values())