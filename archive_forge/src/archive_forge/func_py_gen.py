import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def py_gen(rmin, rmax, nr):
    a = np.linspace(rmin, rmax, nr)
    yield a[0]
    yield a[1]