import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_delitem_slice(self):
    """ Test delitem using a slice.

        This tests suffers from combinatorial explosion, so we parametrize it
        and compare results against the regular list in a quasi fuzzing
        approach.

        """

    def setup(start=10, stop=20):
        rl_ = list(range(start, stop))
        tl_ = List.empty_list(int32)
        for i in range(start, stop):
            tl_.append(i)
        self.assertEqual(rl_, list(tl_))
        return (rl_, tl_)
    start_range = list(range(-20, 30))
    stop_range = list(range(-20, 30))
    step_range = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    rl, tl = setup()
    self.assertEqual(rl, list(tl))
    del rl[:]
    del tl[:]
    self.assertEqual(rl, list(tl))
    for sa in start_range:
        rl, tl = setup()
        del rl[sa:]
        del tl[sa:]
        self.assertEqual(rl, list(tl))
    for so in stop_range:
        rl, tl = setup()
        del rl[:so]
        del tl[:so]
        self.assertEqual(rl, list(tl))
    for se in step_range:
        rl, tl = setup()
        del rl[::se]
        del tl[::se]
        self.assertEqual(rl, list(tl))
    for sa, so in product(start_range, stop_range):
        rl, tl = setup()
        del rl[sa:so]
        del tl[sa:so]
        self.assertEqual(rl, list(tl))
    for sa, se in product(start_range, step_range):
        rl, tl = setup()
        del rl[sa::se]
        del tl[sa::se]
        self.assertEqual(rl, list(tl))
    for so, se in product(stop_range, step_range):
        rl, tl = setup()
        del rl[:so:se]
        del tl[:so:se]
        self.assertEqual(rl, list(tl))
    for sa, so, se in product(start_range, stop_range, step_range):
        rl, tl = setup()
        del rl[sa:so:se]
        del tl[sa:so:se]
        self.assertEqual(rl, list(tl))