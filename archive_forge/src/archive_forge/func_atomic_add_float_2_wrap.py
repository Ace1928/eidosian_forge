import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_add_float_2_wrap(ary):
    atomic_binary_2dim_shared(ary, 1.0, float32, (4, 8), cuda.atomic.add, atomic_cast_none, True)