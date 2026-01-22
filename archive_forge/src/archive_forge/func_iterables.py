import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def iterables(self):
    yield self.listimpl([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
    a = np.linspace(-10, 2, 23)
    yield self.listimpl(a)
    yield self.listimpl(a[::-1])
    self.rnd.shuffle(a)
    yield self.listimpl(a)