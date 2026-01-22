import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heappush_basic(self):
    pyfunc_push = heappush
    cfunc_push = jit(nopython=True)(pyfunc_push)
    pyfunc_pop = heappop
    cfunc_pop = jit(nopython=True)(pyfunc_pop)
    for iterable in self.iterables():
        expected = sorted(iterable)
        heap = self.listimpl([iterable.pop(0)])
        for value in iterable:
            cfunc_push(heap, value)
        got = [cfunc_pop(heap) for _ in range(len(heap))]
        self.assertPreciseEqual(expected, got)