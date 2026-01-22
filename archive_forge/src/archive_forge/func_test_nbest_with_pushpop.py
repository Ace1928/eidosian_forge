import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_nbest_with_pushpop(self):
    pyfunc_heappushpop = heappushpop
    cfunc_heappushpop = jit(nopython=True)(pyfunc_heappushpop)
    pyfunc_heapify = heapify
    cfunc_heapify = jit(nopython=True)(pyfunc_heapify)
    values = np.arange(2000, dtype=np.float64)
    data = self.listimpl(self.rnd.choice(values, 1000))
    heap = data[:10]
    cfunc_heapify(heap)
    for item in data[10:]:
        cfunc_heappushpop(heap, item)
    self.assertPreciseEqual(list(self.heapiter(list(heap))), sorted(data)[-10:])