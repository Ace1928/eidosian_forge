import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_inc2_64(ary, op2):
    atomic_binary_2dim_shared(ary, op2, uint64, (4, 8), cuda.atomic.inc, atomic_cast_none, False)