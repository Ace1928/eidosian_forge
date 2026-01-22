import multiprocessing as mp
import logging
import traceback
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import (skip_on_cudasim, skip_with_cuda_python,
from numba.tests.support import linux_only
def kernel_thread(n):
    f[n_blocks, n_threads, stream](rs[n], xs[n])