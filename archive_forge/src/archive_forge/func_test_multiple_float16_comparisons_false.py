import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
@skip_unless_cc_53
def test_multiple_float16_comparisons_false(self):
    functions = (test_multiple_hcmp_1, test_multiple_hcmp_2, test_multiple_hcmp_3, test_multiple_hcmp_4, test_multiple_hcmp_5)
    for fn in functions:
        with self.subTest(fn=fn):
            compiled = cuda.jit('void(b1[:], f2, f2, f2)')(fn)
            ary = np.zeros(1, dtype=np.bool8)
            arg1 = np.float16(2.0)
            arg2 = np.float16(3.0)
            arg3 = np.float16(1.0)
            compiled[1, 1](ary, arg1, arg2, arg3)
            self.assertFalse(ary[0])