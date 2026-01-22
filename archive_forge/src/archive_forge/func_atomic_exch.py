import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_exch(ary, idx, op2):
    atomic_binary_1dim_shared2(ary, idx, op2, uint32, 32, cuda.atomic.exch, atomic_cast_none)