import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_string_literal_in_ctor(self):

    @jit(nopython=True)
    def foo():
        return Rect(10, 'somestring')
    r = foo()
    self.assertEqual(r, Rect(width=10, height='somestring'))