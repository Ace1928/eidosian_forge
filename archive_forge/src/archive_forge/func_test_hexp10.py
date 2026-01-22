import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_unless_cc_53
def test_hexp10(self):

    @cuda.jit()
    def hexp10_vectors(r, x):
        i = cuda.grid(1)
        if i < len(r):
            r[i] = cuda.fp16.hexp10(x[i])
    N = 32
    np.random.seed(1)
    x = np.random.rand(N).astype(np.float16)
    r = np.zeros_like(x)
    hexp10_vectors[1, N](r, x)
    np.testing.assert_allclose(r, 10 ** x)