from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_reflect_insert(self):
    """make sure list.insert() doesn't crash for refcounted objects (see #7553)
        """

    def pyfunc(con):
        con.insert(1, np.arange(4).astype(np.intp))
    self._check_element_equal(pyfunc)