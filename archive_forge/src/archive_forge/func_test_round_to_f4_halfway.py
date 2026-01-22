import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_round_to_f4_halfway(self):
    compiled = cuda.jit('void(float32[:], float32, int32)')(simple_round_to)
    ary = np.zeros(1, dtype=np.float32)
    val = 0.3425
    ndigits = 3
    compiled[1, 1](ary, val, ndigits)
    self.assertPreciseEqual(ary[0], round(val, ndigits), prec='single')