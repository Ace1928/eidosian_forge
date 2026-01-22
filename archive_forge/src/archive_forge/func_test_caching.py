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
def test_caching(self):
    self.check_pycache(0)
    mod = self.import_module()
    self.check_pycache(0)
    f = mod.add_usecase
    self.assertPreciseEqual(f(2, 3), 6)
    self.check_pycache(2)
    self.assertPreciseEqual(f(2.5, 3), 6.5)
    self.check_pycache(3)
    self.check_hits(f.func, 0, 2)
    f = mod.record_return_aligned
    rec = f(mod.aligned_arr, 1)
    self.assertPreciseEqual(tuple(rec), (2, 43.5))
    f = mod.record_return_packed
    rec = f(mod.packed_arr, 1)
    self.assertPreciseEqual(tuple(rec), (2, 43.5))
    self.check_pycache(6)
    self.check_hits(f.func, 0, 2)
    self.run_in_separate_process()