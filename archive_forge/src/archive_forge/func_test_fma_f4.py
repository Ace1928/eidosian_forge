import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_fma_f4(self):
    compiled = cuda.jit('void(f4[:], f4, f4, f4)')(simple_fma)
    ary = np.zeros(1, dtype=np.float32)
    compiled[1, 1](ary, 2.0, 3.0, 4.0)
    np.testing.assert_allclose(ary[0], 2 * 3 + 4)