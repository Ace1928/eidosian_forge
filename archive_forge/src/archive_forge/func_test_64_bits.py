import numpy as np
from numba import njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import TestCase
def test_64_bits(self):
    dtypes = (np.uint64, np.int64, np.float64)
    inputs = ((1, np.uint64, (1, 1, 5e-324)), (-1, np.int64, (18446744073709551615, -1, np.nan)), (1.0, np.float64, (4607182418800017408, 4607182418800017408, 1.0)))
    self.do_testing(inputs, dtypes)