from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_nanquantile_basic(self):
    pyfunc = array_nanquantile_global
    self.check_percentile_and_quantile(pyfunc, q_upper_bound=1)
    self.check_percentile_edge_cases(pyfunc, q_upper_bound=1)
    self.check_quantile_exceptions(pyfunc)