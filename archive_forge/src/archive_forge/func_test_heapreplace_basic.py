import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heapreplace_basic(self):
    pyfunc = heapreplace
    cfunc = jit(nopython=True)(pyfunc)
    a = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]
    heapify(a)
    b = self.listimpl(a)
    for item in [-4, 4, 14]:
        pyfunc(a, item)
        cfunc(b, item)
        self.assertPreciseEqual(a, list(b))
    a = np.linspace(-3, 13, 20)
    a[4] = np.nan
    a[-1] = np.inf
    a = a.tolist()
    heapify(a)
    b = self.listimpl(a)
    for item in [-4.0, 3.142, -np.inf, np.inf]:
        pyfunc(a, item)
        cfunc(b, item)
        self.assertPreciseEqual(a, list(b))