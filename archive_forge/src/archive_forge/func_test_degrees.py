import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_degrees(self):
    pyfunc = degrees
    x_types = [types.int16, types.int32, types.int64, types.uint16, types.uint32, types.uint64, types.float32, types.float64]
    x_values = [1, 1, 1, 1, 1, 1, 1.0, 1.0]
    self.run_unary(pyfunc, x_types, x_values)