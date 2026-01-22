import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_round_f8(self):
    compiled = cuda.jit('void(int64[:], float64)')(simple_round)
    ary = np.zeros(1, dtype=np.int64)
    for i in [-3.0, -2.5, -2.25, -1.5, 1.5, 2.25, 2.5, 2.75]:
        compiled[1, 1](ary, i)
        self.assertEqual(ary[0], round(i))