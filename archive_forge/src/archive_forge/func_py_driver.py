import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def py_driver(args):
    rmin, rmax, nr = args
    points = np.empty(nr, dtype=np.complex128)
    for i, c in enumerate(py_gen(rmin, rmax, nr)):
        points[i] = c
    return points