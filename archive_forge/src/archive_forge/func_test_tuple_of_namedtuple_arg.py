import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_tuple_of_namedtuple_arg(self):
    Point = namedtuple('Point', ('x', 'y', 'z'))
    a = (Point(x=1.0, y=2.0, z=3.0), Point(x=4.0, y=5.0, z=6.0))
    b = (Point(x=1.5, y=2.5, z=3.5), Point(x=4.5, y=5.5, z=6.5))
    self.check_tuple_arg(a, b)