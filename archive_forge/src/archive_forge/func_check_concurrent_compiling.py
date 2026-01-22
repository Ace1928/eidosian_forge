import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
def check_concurrent_compiling():

    @cuda.jit
    def foo(x):
        x[0] += 1

    def use_foo(x):
        foo[1, 1](x)
        return x
    arrays = [cuda.to_device(np.arange(10)) for i in range(10)]
    expected = np.arange(10)
    expected[0] += 1
    with ThreadPoolExecutor(max_workers=4) as e:
        for ary in e.map(use_foo, arrays):
            np.testing.assert_equal(ary, expected)