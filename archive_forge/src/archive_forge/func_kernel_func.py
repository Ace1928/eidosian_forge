import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def kernel_func(out, a, b):
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = device_func(a[i], b[i])