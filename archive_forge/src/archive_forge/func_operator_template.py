import numpy as np
from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
from numba import cuda
from numba.core.types import f2, b1
from numba.cuda import compile_ptx
import operator
import itertools
from numba.np.numpy_support import from_dtype
def operator_template(self, op):

    @cuda.jit
    def foo(a, b):
        i = 0
        a[i] = op(a[i], b[i])
    a = np.ones(1)
    b = np.ones(1)
    res = a.copy()
    foo[1, 1](res, b)
    np.testing.assert_equal(res, op(a, b))