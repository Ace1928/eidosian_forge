import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def check_predicate_func(self, pyfunc):
    self.run_unary(pyfunc, [types.boolean(tp) for tp in (types.complex128, types.complex64)], self.basic_values())