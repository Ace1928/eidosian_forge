import numpy as np
from numba import cuda, int32, int64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.core import config
def use_vote_sync_any(ary_in, ary_out):
    i = cuda.grid(1)
    pred = cuda.any_sync(4294967295, ary_in[i])
    ary_out[i] = pred