import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
@skip_parfors_unsupported
def test_issue7380(self):

    @njit(inline='always')
    def bar(x):
        for i in range(x.size):
            x[i] += 1

    @njit(parallel=True)
    def foo(a):
        for i in prange(a.shape[0]):
            bar(a[i])
    a = np.ones((10, 10))
    foo(a)
    self.assertPreciseEqual(a, 2 * np.ones_like(a))

    @njit(parallel=True)
    def foo_bad(a):
        for i in prange(a.shape[0]):
            x = a[i]
            for i in range(x.size):
                x[i] += 1
    with self.assertRaises(errors.UnsupportedRewriteError) as e:
        foo_bad(a)
    self.assertIn('Overwrite of parallel loop index', str(e.exception))