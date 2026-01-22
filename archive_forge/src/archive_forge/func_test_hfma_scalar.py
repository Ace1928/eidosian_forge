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
def test_hfma_scalar(self):
    compiled = cuda.jit('void(f2[:], f2, f2, f2)')(simple_hfma_scalar)
    ary = np.zeros(1, dtype=np.float16)
    arg1 = np.float16(2.0)
    arg2 = np.float16(3.0)
    arg3 = np.float16(4.0)
    compiled[1, 1](ary, arg1, arg2, arg3)
    ref = arg1 * arg2 + arg3
    np.testing.assert_allclose(ary[0], ref)