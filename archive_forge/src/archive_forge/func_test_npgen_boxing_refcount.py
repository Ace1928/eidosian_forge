import numba
import numpy as np
import sys
import itertools
import gc
from numba import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np.random.generator_methods import _get_proper_func
from numba.np.random.generator_core import next_uint32, next_uint64, next_double
from numpy.random import MT19937, Generator
from numba.core.errors import TypingError
from numba.tests.support import run_in_new_process_caching, SerialMixin
def test_npgen_boxing_refcount(self):
    rng_instance = np.random.default_rng()
    no_box = numba.njit(lambda x: x.random())
    do_box = numba.njit(lambda x: x)
    y = do_box(rng_instance)
    gc.collect()
    ref_1 = sys.getrefcount(rng_instance)
    del y
    no_box(rng_instance)
    gc.collect()
    ref_2 = sys.getrefcount(rng_instance)
    self.assertEqual(ref_1, ref_2 + 1)