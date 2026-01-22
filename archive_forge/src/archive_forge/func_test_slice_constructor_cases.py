from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def test_slice_constructor_cases(self):
    """
        Test that slice constructor behaves same in python and compiled code.
        """
    options = (None, -1, 0, 1)
    arg_cases = chain.from_iterable((product(options, repeat=n) for n in range(5)))
    array = np.arange(10)
    cfunc = jit(nopython=True)(slice_construct_and_use)
    self.disable_leak_check()
    for args in arg_cases:
        try:
            expected = slice_construct_and_use(args, array)
        except TypeError as py_type_e:
            n_args = len(args)
            self.assertRegex(str(py_type_e), 'slice expected at (most|least) (3|1) arguments?, got {}'.format(n_args))
            with self.assertRaises(TypingError) as numba_e:
                cfunc(args, array)
            self.assertIn(_header_lead, str(numba_e.exception))
            self.assertIn(', '.join((str(typeof(arg)) for arg in args)), str(numba_e.exception))
        except Exception as py_e:
            with self.assertRaises(type(py_e)) as numba_e:
                cfunc(args, array)
            self.assertIn(str(py_e), str(numba_e.exception))
        else:
            self.assertPreciseEqual(expected, cfunc(args, array))