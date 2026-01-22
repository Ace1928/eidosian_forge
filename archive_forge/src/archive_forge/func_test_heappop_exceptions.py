import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_heappop_exceptions(self):
    pyfunc = heappop
    cfunc = jit(nopython=True)(pyfunc)
    self.disable_leak_check()
    with self.assertTypingError() as e:
        cfunc((1, 5, 4))
    msg = 'heap argument must be a list'
    self.assertIn(msg, str(e.exception))