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
def test_run_timsort_with_values(self):
    f = self.timsort.run_timsort_with_values
    for size_factor in (1, 5):
        chunk_size = 80 * size_factor
        a = self.dupsorted_list(chunk_size)
        b = self.duprandom_list(chunk_size)
        c = self.revsorted_list(chunk_size)
        orig_keys = a + b + c
        orig_values = list(range(1000, 1000 + len(orig_keys)))
        keys = self.array_factory(orig_keys)
        values = self.array_factory(orig_values)
        f(keys, values)
        self.assertSortedValues(orig_keys, orig_values, keys, values)