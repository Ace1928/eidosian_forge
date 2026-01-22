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
def test_run_quicksort(self):
    f = self.quicksort.run_quicksort
    for size_factor in (1, 5):
        sizes = (15, 20)
        all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
        for chunks in itertools.product(*all_lists):
            orig_keys = sum(chunks, [])
            shape_list = self.get_shapes(len(orig_keys))
            shape_list.append(None)
            for shape in shape_list:
                keys = self.array_factory(orig_keys, shape=shape)
                keys_copy = self.array_factory(orig_keys, shape=shape)
                f(keys)
                keys_copy.sort()
                self.assertSorted(keys_copy, keys)