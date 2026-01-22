import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_allocation_mt(self):
    """
        This test exercises the array allocation in multithreaded usecase.
        This stress the freelist inside NRT.
        """

    def pyfunc(inp):
        out = np.empty(inp.size)
        for i in range(out.size):
            out[i] = 0
        for i in range(inp[0]):
            tmp = np.empty(inp.size)
            for j in range(tmp.size):
                tmp[j] = inp[j]
            for j in range(tmp.size):
                out[j] += tmp[j] + i
        return out
    cfunc = nrtjit(pyfunc)
    size = 10
    arr = np.random.randint(1, 10, size)
    frozen_arr = arr.copy()
    np.testing.assert_equal(pyfunc(arr), cfunc(arr))
    np.testing.assert_equal(frozen_arr, arr)
    workers = []
    inputs = []
    outputs = []

    def wrapped(inp, out):
        out[:] = cfunc(inp)
    for i in range(100):
        arr = np.random.randint(1, 10, size)
        out = np.empty_like(arr)
        thread = threading.Thread(target=wrapped, args=(arr, out), name='worker{0}'.format(i))
        workers.append(thread)
        inputs.append(arr)
        outputs.append(out)
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    for inp, out in zip(inputs, outputs):
        np.testing.assert_equal(pyfunc(inp), out)