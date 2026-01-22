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
def check_sorted_list(l):
    l = self.array_factory(l)
    for key in (l[5], l[15], l[0], -1000, l[-1], 1000):
        check_all_hints(l, key, 0, n)
        check_all_hints(l, key, 1, n - 1)
        check_all_hints(l, key, 8, n - 8)