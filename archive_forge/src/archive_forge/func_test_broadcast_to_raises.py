from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_broadcast_to_raises(self):
    pyfunc = numpy_broadcast_to
    cfunc = jit(nopython=True)(pyfunc)
    data = [[np.zeros((0,)), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [np.zeros((1,)), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [np.zeros((3,)), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [(), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [(123,), (), TypingError, 'Cannot broadcast a non-scalar to a scalar array'], [np.zeros((3,)), (1,), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((3,)), (2,), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((3,)), (4,), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((1, 2)), (2, 1), ValueError, 'operands could not be broadcast together with remapped shapes'], [np.zeros((1, 1)), (1,), ValueError, 'input operand has more dimensions than allowed by the axis remapping'], [np.zeros((2, 2)), (3,), ValueError, 'input operand has more dimensions than allowed by the axis remapping'], [np.zeros((1,)), -1, ValueError, 'all elements of broadcast shape must be non-negative'], [np.zeros((1,)), (-1,), ValueError, 'all elements of broadcast shape must be non-negative'], [np.zeros((1, 2)), (-1, 2), ValueError, 'all elements of broadcast shape must be non-negative'], [np.zeros((1, 2)), (1.1, 2.2), TypingError, 'The second argument "shape" must be a tuple of integers'], ['hello', (3,), TypingError, 'The first argument "array" must be array-like'], [3, (2, 'a'), TypingError, 'object cannot be interpreted as an integer']]
    self.disable_leak_check()
    for arr, target_shape, err, msg in data:
        with self.assertRaises(err) as raises:
            cfunc(arr, target_shape)
        self.assertIn(msg, str(raises.exception))