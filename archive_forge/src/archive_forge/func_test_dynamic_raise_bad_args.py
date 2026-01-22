import numpy as np
import sys
import traceback
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import (TestCase, expected_failure_py311,
import unittest
def test_dynamic_raise_bad_args(self):

    def raise_literal_dict():
        raise ValueError({'a': 1, 'b': np.ones(4)})

    def raise_range():
        raise ValueError(range(3))

    def raise_rng(rng):
        raise ValueError(rng.bit_generator)
    funcs = [(raise_literal_dict, ()), (raise_range, ()), (raise_rng, (types.npy_rng,))]
    for pyfunc, argtypes in funcs:
        msg = '.*Cannot convert native .* to a Python object.*'
        with self.assertRaisesRegex(errors.TypingError, msg):
            njit(argtypes)(pyfunc)