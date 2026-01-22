import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_xor_global_2(ary, op2):
    atomic_binary_2dim_global(ary, op2, cuda.atomic.xor, atomic_cast_none, False)