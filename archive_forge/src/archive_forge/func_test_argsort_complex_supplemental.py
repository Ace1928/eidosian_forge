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
def test_argsort_complex_supplemental(self):

    def check(pyfunc, is_stable):
        cfunc = jit(nopython=True)(pyfunc)
        for real in self.float_arrays():
            imag = real[:]
            np.random.shuffle(imag)
            orig = np.array([complex(*x) for x in zip(real, imag)])
            self.check_argsort(pyfunc, cfunc, orig, dict(is_stable=is_stable))
    check(argsort_kind_usecase, is_stable=True)
    check(np_argsort_kind_usecase, is_stable=True)
    check(argsort_kind_usecase, is_stable=False)
    check(np_argsort_kind_usecase, is_stable=False)