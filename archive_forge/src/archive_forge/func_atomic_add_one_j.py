import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
@cuda.jit
def atomic_add_one_j(values):
    i = cuda.grid(1)
    cuda.atomic.add(values.imag, i, 1)