from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_nptimedelta(self, pyfunc):
    arr = np.arange(10).astype(dtype='m8[s]')
    self._do_check_nptimedelta(pyfunc, arr)