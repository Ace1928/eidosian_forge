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
def test_many_locals(self):
    self.check_pycache(0)
    mod = self.import_module()
    f = mod.many_locals
    f[1, 1]()
    self.check_pycache(2)