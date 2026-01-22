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
def test_no_caching(self):
    mod = self.import_module()
    f = mod.add_nocache_usecase
    self.assertPreciseEqual(f(2, 3), 6)
    self.check_pycache(0)