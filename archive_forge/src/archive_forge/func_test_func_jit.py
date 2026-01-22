from __future__ import print_function, absolute_import, division
import sys
import os
import re
import multiprocessing
import unittest
import numpy as np
from numba import (njit, set_num_threads, get_num_threads, prange, config,
from numba.np.ufunc.parallel import get_thread_id
from numba.core.errors import TypingError
from numba.tests.support import TestCase, skip_parfors_unsupported, tag
from numba.tests.test_parallel_backend import TestInSubprocess
@njit(parallel=True)
def test_func_jit(nthreads):
    set_num_threads(nthreads)
    lens = np.zeros(nthreads)
    total = 0
    for i in prange(nthreads):
        my_acc, tids = work(nthreads + 1)
        lens[i] = len(tids)
        total += my_acc
    return (total, np.unique(lens))