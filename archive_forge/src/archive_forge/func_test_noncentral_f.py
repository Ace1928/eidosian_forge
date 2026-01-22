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
def test_noncentral_f(self):
    test_sizes = [None, (), (100,), (10, 20, 30)]
    bitgen_types = [None, MT19937]
    dist_func = lambda x, size, dtype: x.noncentral_f(3.0, 20.0, 3.0, size=size)
    for _size, _bitgen in itertools.product(test_sizes, bitgen_types):
        with self.subTest(_size=_size, _bitgen=_bitgen):
            self.check_numpy_parity(dist_func, _bitgen, None, _size, None, adjusted_ulp_prec)
    dist_func = lambda x, dfnum, dfden, nonc, size: x.noncentral_f(dfnum=dfnum, dfden=dfden, nonc=nonc, size=size)
    valid_args = [3.0, 5.0, 3.0, (1,)]
    self._check_invalid_types(dist_func, ['dfnum', 'dfden', 'nonc', 'size'], valid_args, ['x', 'x', 'x', ('x',)])
    rng = np.random.default_rng()
    valid_args = [rng] + valid_args
    nb_dist_func = numba.njit(dist_func)
    with self.assertRaises(ValueError) as raises:
        curr_args = valid_args.copy()
        curr_args[1] = 0
        nb_dist_func(*curr_args)
    self.assertIn('dfnum <= 0', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        curr_args = valid_args.copy()
        curr_args[2] = 0
        nb_dist_func(*curr_args)
    self.assertIn('dfden <= 0', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        curr_args = valid_args.copy()
        curr_args[3] = -1
        nb_dist_func(*curr_args)
    self.assertIn('nonc < 0', str(raises.exception))
    self.disable_leak_check()