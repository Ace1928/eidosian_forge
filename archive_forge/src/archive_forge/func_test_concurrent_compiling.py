import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
@unittest.skipIf(not has_concurrent_futures, 'no concurrent.futures')
def test_concurrent_compiling(self):
    check_concurrent_compiling()