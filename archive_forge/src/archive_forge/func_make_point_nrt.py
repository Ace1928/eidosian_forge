import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def make_point_nrt(n):
    r = Rect(list(range(n)), np.zeros(n + 1))
    p = Point(r, len(r.width), len(r.height))
    return p