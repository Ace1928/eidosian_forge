import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_sub_double_global_2(ary):
    atomic_binary_2dim_global(ary, 1.0, cuda.atomic.sub, atomic_cast_none, False)