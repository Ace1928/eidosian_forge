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
def check_numpy_parity(self, distribution_func, bitgen_type=None, seed=None, test_size=None, test_dtype=None, ulp_prec=5):
    distribution_func = numba.njit(distribution_func)
    if seed is None:
        seed = 1
    if bitgen_type is None:
        numba_rng_instance = np.random.default_rng(seed=seed)
        numpy_rng_instance = np.random.default_rng(seed=seed)
    else:
        numba_rng_instance = Generator(bitgen_type(seed))
        numpy_rng_instance = Generator(bitgen_type(seed))
    numba_res = distribution_func(numba_rng_instance, test_size, test_dtype)
    numpy_res = distribution_func.py_func(numpy_rng_instance, test_size, test_dtype)
    if isinstance(numba_res, np.ndarray) and np.issubdtype(numba_res.dtype, np.floating) or isinstance(numba_res, float):
        np.testing.assert_array_max_ulp(numpy_res, numba_res, maxulp=ulp_prec, dtype=test_dtype)
    else:
        np.testing.assert_equal(numba_res, numpy_res)
    numba_gen_state = numba_rng_instance.__getstate__()['state']
    numpy_gen_state = numpy_rng_instance.__getstate__()['state']
    for _state_key in numpy_gen_state:
        self.assertPreciseEqual(numba_gen_state[_state_key], numpy_gen_state[_state_key])