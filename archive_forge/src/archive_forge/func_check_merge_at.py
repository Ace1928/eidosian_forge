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
def check_merge_at(self, a, b):
    f = self.timsort.merge_at
    na = len(a)
    nb = len(b)
    orig_keys = [42] + a + b + [-42]
    ssa = 1
    ssb = ssa + na
    stack_sentinel = MergeRun(-42, -42)

    def run_merge_at(ms, keys, i):
        new_ms = f(ms, keys, keys, i)
        self.assertEqual(keys[0], orig_keys[0])
        self.assertEqual(keys[-1], orig_keys[-1])
        self.assertSorted(orig_keys[1:-1], keys[1:-1])
        self.assertIs(new_ms.pending, ms.pending)
        self.assertEqual(ms.pending[i], (ssa, na + nb))
        self.assertEqual(ms.pending[0], stack_sentinel)
        return new_ms
    keys = self.array_factory(orig_keys)
    ms = self.merge_init(keys)
    ms = self.timsort.merge_append(ms, stack_sentinel)
    i = ms.n
    ms = self.timsort.merge_append(ms, MergeRun(ssa, na))
    ms = self.timsort.merge_append(ms, MergeRun(ssb, nb))
    ms = run_merge_at(ms, keys, i)
    self.assertEqual(ms.n, i + 1)
    keys = self.array_factory(orig_keys)
    ms = self.merge_init(keys)
    ms = self.timsort.merge_append(ms, stack_sentinel)
    i = ms.n
    ms = self.timsort.merge_append(ms, MergeRun(ssa, na))
    ms = self.timsort.merge_append(ms, MergeRun(ssb, nb))
    last_run = MergeRun(ssb + nb, 1)
    ms = self.timsort.merge_append(ms, last_run)
    ms = run_merge_at(ms, keys, i)
    self.assertEqual(ms.n, i + 2)
    self.assertEqual(ms.pending[ms.n - 1], last_run)