from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_unconfigured_untyped_cudakernel(self):
    kernfunc = cuda.jit(noop)
    self._test_unconfigured(kernfunc)