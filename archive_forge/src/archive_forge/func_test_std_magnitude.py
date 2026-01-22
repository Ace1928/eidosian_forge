from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_std_magnitude(self):
    self.check_aggregation_magnitude(array_std)
    self.check_aggregation_magnitude(array_std_global)