import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def gen4(x, y, z):
    for i in range(3):
        yield z
        yield (y + z)
    return
    yield x