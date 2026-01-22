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
def test_hdiv(self):
    compiled = cuda.jit('void(f2[:], f2[:], f2[:])')(simple_hdiv_kernel)
    arry1 = np.random.randint(-65504, 65505, size=500).astype(np.float16)
    arry2 = np.random.randint(-65504, 65505, size=500).astype(np.float16)
    ary = np.zeros_like(arry1, dtype=np.float16)
    compiled.forall(ary.size)(ary, arry1, arry2)
    ref = arry1 / arry2
    np.testing.assert_allclose(ary, ref)