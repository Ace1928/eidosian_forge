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
def test_randomgen_caching(self):
    nb_rng = np.random.default_rng(1)
    np_rng = np.random.default_rng(1)
    numba_func = numba.njit(lambda x: x.random(10), cache=True)
    self.assertPreciseEqual(np_rng.random(10), numba_func(nb_rng))
    self.assertPreciseEqual(np_rng.random(10), numba_func(nb_rng))
    res = run_in_new_process_caching(test_generator_caching)
    self.assertEqual(res['exitcode'], 0)