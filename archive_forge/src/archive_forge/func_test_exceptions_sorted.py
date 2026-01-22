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
def test_exceptions_sorted(self):

    @njit
    def foo_sorted(x, key=None, reverse=False):
        return sorted(x[:], key=key, reverse=reverse)

    @njit
    def foo_sort(x, key=None, reverse=False):
        new_x = x[:]
        new_x.sort(key=key, reverse=reverse)
        return new_x

    @njit
    def external_key(z):
        return 1.0 / z
    a = [3, 1, 4, 1, 5, 9]
    for impl in (foo_sort, foo_sorted):
        with self.assertRaises(errors.TypingError) as raises:
            impl(a, key='illegal')
        expect = 'Key must be None or a Numba JIT compiled function'
        self.assertIn(expect, str(raises.exception))
        with self.assertRaises(errors.TypingError) as raises:
            impl(a, key=external_key, reverse='go backwards')
        expect = "an integer is required for 'reverse'"
        self.assertIn(expect, str(raises.exception))