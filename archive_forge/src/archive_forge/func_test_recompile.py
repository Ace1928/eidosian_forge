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
def test_recompile(self):
    mod = self.import_module()
    f = mod.add_usecase
    self.assertPreciseEqual(f(2, 3), 6)
    mod = self.import_module()
    f = mod.add_usecase
    mod.Z = 10
    self.assertPreciseEqual(f(2, 3), 6)
    f.func.recompile()
    self.assertPreciseEqual(f(2, 3), 15)
    mod = self.import_module()
    f = mod.add_usecase
    self.assertPreciseEqual(f(2, 3), 15)