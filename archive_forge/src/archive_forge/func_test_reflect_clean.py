import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_reflect_clean(self):
    """
        When the set wasn't mutated, no reflection should take place.
        """
    cfunc = jit(nopython=True)(noop)
    s = set([12.5j])
    ids = [id(x) for x in s]
    cfunc(s)
    self.assertEqual([id(x) for x in s], ids)