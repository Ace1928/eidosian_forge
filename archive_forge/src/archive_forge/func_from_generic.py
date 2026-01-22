from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def from_generic(pyfuncs_to_use):
    """Decorator for generic check functions.
        Iterates over 'pyfuncs_to_use', calling 'func' with the iterated
        item as first argument. Example:

        @from_generic(numpy_array_reshape, array_reshape)
        def check_only_shape(pyfunc, arr, shape, expected_shape):
            # Only check Numba result to avoid Numpy bugs
            self.memory_leak_setup()
            got = generic_run(pyfunc, arr, shape)
            self.assertEqual(got.shape, expected_shape)
            self.assertEqual(got.size, arr.size)
            del got
            self.memory_leak_teardown()
    """

    def decorator(func):

        def result(*args, **kwargs):
            return [func(pyfunc, *args, **kwargs) for pyfunc in pyfuncs_to_use]
        return result
    return decorator