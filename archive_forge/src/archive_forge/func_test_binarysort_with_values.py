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
def test_binarysort_with_values(self):
    n = 20
    v = list(range(100, 100 + n))

    def check(l, n, start=0):
        res = self.array_factory(l)
        res_v = self.array_factory(v)
        f(res, res_v, 0, n, start)
        self.assertSortedValues(l, v, res, res_v)
    f = self.timsort.binarysort
    l = self.sorted_list(n)
    check(l, n)
    check(l, n, n // 2)
    l = self.revsorted_list(n)
    check(l, n)
    l = self.initially_sorted_list(n, n // 2)
    check(l, n)
    check(l, n, n // 2)
    l = self.revsorted_list(n)
    check(l, n)
    l = self.random_list(n)
    check(l, n)
    l = self.duprandom_list(n)
    check(l, n)