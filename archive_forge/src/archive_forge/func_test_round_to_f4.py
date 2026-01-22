import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_round_to_f4(self):
    compiled = cuda.jit('void(float32[:], float32, int32)')(simple_round_to)
    ary = np.zeros(1, dtype=np.float32)
    np.random.seed(123)
    vals = np.random.random(32).astype(np.float32)
    np.concatenate((vals, np.array([np.inf, -np.inf, np.nan])))
    digits = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 13)
    for val, ndigits in itertools.product(vals, digits):
        with self.subTest(val=val, ndigits=ndigits):
            compiled[1, 1](ary, val, ndigits)
            self.assertPreciseEqual(ary[0], round(val, ndigits), prec='single')