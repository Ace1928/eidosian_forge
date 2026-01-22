import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_simple_gridsize2d(self):
    compiled = cuda.jit('void(int32[::1])')(simple_gridsize2d)
    ntid = (4, 3)
    nctaid = (5, 6)
    ary = np.zeros(2, dtype=np.int32)
    compiled[nctaid, ntid](ary)
    self.assertEqual(ary[0], nctaid[0] * ntid[0])
    self.assertEqual(ary[1], nctaid[1] * ntid[1])