import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_namedtuple_of_array_arg(self):
    xs1 = np.arange(10, dtype=np.int32)
    ys1 = xs1 + 2
    xs2 = np.arange(10, dtype=np.int32) * 2
    ys2 = xs2 + 1
    Points = namedtuple('Points', ('xs', 'ys'))
    a = Points(xs=xs1, ys=ys1)
    b = Points(xs=xs2, ys=ys2)
    self.check_tuple_arg(a, b)