from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_shapes_invalid_type(self):
    pyfunc = numpy_broadcast_shapes
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    inps = [((1, 2), ('hello',)), (3.4,), ('string',), (1.2, 'a'), (1, (1.2, 'a'))]
    for inp in inps:
        with self.assertRaises(TypingError) as raises:
            cfunc(*inp)
        self.assertIn('must be either an int or tuple[int]', str(raises.exception))