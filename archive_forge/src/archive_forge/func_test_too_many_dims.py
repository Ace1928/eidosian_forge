from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_too_many_dims(self):
    kernfunc = cuda.jit(noop)
    with self.assertRaises(ValueError) as raises:
        kernfunc[(1, 2, 3, 4), (5, 6)]
    self.assertIn('griddim must be a sequence of 1, 2 or 3 integers, got [1, 2, 3, 4]', str(raises.exception))
    with self.assertRaises(ValueError) as raises:
        kernfunc[(1, 2), (3, 4, 5, 6)]
    self.assertIn('blockdim must be a sequence of 1, 2 or 3 integers, got [3, 4, 5, 6]', str(raises.exception))