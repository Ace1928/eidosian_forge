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
def test_gallop_left(self):
    n = 20
    f = self.timsort.gallop_left

    def check(l, key, start, stop, hint):
        k = f(key, l, start, stop, hint)
        self.assertGreaterEqual(k, start)
        self.assertLessEqual(k, stop)
        if k > start:
            self.assertLess(l[k - 1], key)
        if k < stop:
            self.assertGreaterEqual(l[k], key)

    def check_all_hints(l, key, start, stop):
        for hint in range(start, stop):
            check(l, key, start, stop, hint)

    def check_sorted_list(l):
        l = self.array_factory(l)
        for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
            check_all_hints(l, key, 0, n)
            check_all_hints(l, key, 1, n - 1)
            check_all_hints(l, key, 8, n - 8)
    l = self.sorted_list(n, offset=100)
    check_sorted_list(l)
    l = self.dupsorted_list(n, offset=100)
    check_sorted_list(l)