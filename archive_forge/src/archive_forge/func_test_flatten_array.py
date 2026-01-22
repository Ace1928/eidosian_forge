from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_flatten_array(self, flags=enable_pyobj_flags, layout='C'):
    a = np.arange(9).reshape(3, 3)
    if layout == 'F':
        a = a.T
    pyfunc = flatten_array
    arraytype1 = typeof(a)
    if layout == 'A':
        arraytype1 = arraytype1.copy(layout='A')
    self.assertEqual(arraytype1.layout, layout)
    cfunc = jit((arraytype1,), **flags)(pyfunc)
    expected = pyfunc(a)
    got = cfunc(a)
    np.testing.assert_equal(expected, got)