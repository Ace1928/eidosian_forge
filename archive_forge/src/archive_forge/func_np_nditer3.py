import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def np_nditer3(a, b, c):
    res = []
    for u, v, w in np.nditer((a, b, c)):
        res.append((u.item(), v.item(), w.item()))
    return res