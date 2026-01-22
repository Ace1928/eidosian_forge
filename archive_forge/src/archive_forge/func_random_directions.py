import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
@jit(nopython=True)
def random_directions(n):
    for i in range(n):
        vec = np.empty(3)
        vec[:] = 12
        yield vec