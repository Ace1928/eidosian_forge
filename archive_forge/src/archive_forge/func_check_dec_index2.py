import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def check_dec_index2(self, ary, idx, rconst, sig, nblocks, blksize, func):
    orig = ary.copy()
    cuda_func = cuda.jit(sig)(func)
    cuda_func[nblocks, blksize](idx, ary, rconst)
    np.testing.assert_equal(ary, np.where(orig == 0, rconst, np.where(orig > rconst, rconst, orig - 1)))