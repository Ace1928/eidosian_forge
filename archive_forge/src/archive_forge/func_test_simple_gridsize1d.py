import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def test_simple_gridsize1d(self):
    compiled = cuda.jit('void(int32[::1])')(simple_gridsize1d)
    ntid, nctaid = (3, 7)
    ary = np.zeros(1, dtype=np.int32)
    compiled[nctaid, ntid](ary)
    self.assertEqual(ary[0], nctaid * ntid)