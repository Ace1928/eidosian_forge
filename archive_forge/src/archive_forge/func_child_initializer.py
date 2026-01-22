import multiprocessing
import os
import shutil
import unittest
import warnings
from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
def child_initializer():
    from numba.core import config
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    config.CUDA_WARN_ON_IMPLICIT_COPY = 0