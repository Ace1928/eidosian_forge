import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heapify(self):
    pyfunc = heapify
    cfunc = jit(nopython=True)(pyfunc)
    for size in list(range(1, 30)) + [20000]:
        heap = self.listimpl(self.rnd.random_sample(size))
        cfunc(heap)
        self.check_invariant(heap)