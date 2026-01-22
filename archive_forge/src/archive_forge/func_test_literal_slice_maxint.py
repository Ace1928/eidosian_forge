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
def test_literal_slice_maxint(self):

    @njit()
    def foo(z):
        return literally(z)
    maxval = int(2 ** 63)
    with self.assertRaises(ValueError) as e:
        foo(slice(None, None, -maxval - 1))
    self.assertIn('Int value is too large', str(e.exception))