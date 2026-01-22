import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def record_read_2d_array(r, a):
    a[0, 0] = r.j[0, 0]
    a[0, 1] = r.j[0, 1]
    a[1, 0] = r.j[1, 0]
    a[1, 1] = r.j[1, 1]
    a[2, 0] = r.j[2, 0]
    a[2, 1] = r.j[2, 1]