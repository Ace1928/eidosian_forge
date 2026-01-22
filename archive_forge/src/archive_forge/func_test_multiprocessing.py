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
def test_multiprocessing(self):
    mod = self.import_module()
    f = mod.simple_usecase_caller
    n = 3
    try:
        ctx = multiprocessing.get_context('spawn')
    except AttributeError:
        ctx = multiprocessing
    pool = ctx.Pool(n, child_initializer)
    try:
        res = sum(pool.imap(f, range(n)))
    finally:
        pool.close()
    self.assertEqual(res, n * (n - 1) // 2)