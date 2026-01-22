import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
def test_np_sort_float(self):
    pyfunc = np_sort_usecase
    cfunc = jit(nopython=True)(pyfunc)
    for size in (5, 20, 50, 500):
        orig = np.random.random(size=size) * 100
        orig[np.random.random(size=size) < 0.1] = float('nan')
        self.check_sort_copy(pyfunc, cfunc, orig)