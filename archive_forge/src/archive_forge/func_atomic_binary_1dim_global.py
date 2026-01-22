import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
@cuda.jit(device=True)
def atomic_binary_1dim_global(ary, idx, ary_nelements, op2, binop_func, neg_idx):
    tid = cuda.threadIdx.x
    bin = int(idx[tid] % ary_nelements)
    if neg_idx:
        bin = bin - ary_nelements
    binop_func(ary, bin, op2)