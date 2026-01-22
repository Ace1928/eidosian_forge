import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def test_add_discard(self):
    """
        Check that the insertion logic does not create an infinite lookup
        chain with deleted entries (insertion should happen at the first
        deleted entry, not at the free entry at the end of the chain).
        See issue #1913.
        """
    pyfunc = add_discard_usecase
    check = self.unordered_checker(pyfunc)
    a = b = None
    while a == b:
        a, b = self.sparse_array(2)
    check((a,), b, b)