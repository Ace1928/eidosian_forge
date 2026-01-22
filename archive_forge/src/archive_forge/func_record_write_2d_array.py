import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def record_write_2d_array(r):
    r.i = 3
    r.j[0, 0] = 5.0
    r.j[0, 1] = 6.0
    r.j[1, 0] = 7.0
    r.j[1, 1] = 8.0
    r.j[2, 0] = 9.0
    r.j[2, 1] = 10.0