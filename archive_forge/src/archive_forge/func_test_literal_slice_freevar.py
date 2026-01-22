from functools import partial
import itertools
from itertools import chain, product, starmap
import sys
import numpy as np
from numba import jit, literally, njit, typeof, TypingError
from numba.core import utils, types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.types.functions import _header_lead
import unittest
def test_literal_slice_freevar(self):
    z = slice(1, 2, 3)

    @njit
    def foo():
        return z
    self.assertEqual(z, foo())