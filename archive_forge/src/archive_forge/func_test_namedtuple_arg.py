import numpy as np
from collections import namedtuple
from itertools import product
from numba import vectorize
from numba import cuda, int32, float32, float64
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.testing import skip_on_cudasim
from numba.cuda.testing import CUDATestCase
import unittest
def test_namedtuple_arg(self):
    Point = namedtuple('Point', ('x', 'y', 'z'))
    a = Point(x=1.0, y=2.0, z=3.0)
    b = Point(x=4.0, y=5.0, z=6.0)
    self.check_tuple_arg(a, b)