import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def tuple_unpack_static_getitem_err():
    a, b, c, d = ([], [], [], 0.0)
    a.append(1)
    b.append(1)
    return