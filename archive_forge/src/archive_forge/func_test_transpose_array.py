from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_transpose_array(self, flags=enable_pyobj_flags):

    @from_generic([transpose_array, numpy_transpose_array])
    def check(pyfunc):
        a = np.arange(9).reshape(3, 3)
        arraytype1 = typeof(a)
        cfunc = jit((arraytype1,), **flags)(pyfunc)
        expected = pyfunc(a)
        got = cfunc(a)
        np.testing.assert_equal(expected, got)
    check()