import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
def test_numpy_allocators(self):
    fns = [np.ones, np.zeros]
    for fn in fns:
        with self.subTest(fn.__name__):
            self.check_numpy_allocators(fn)