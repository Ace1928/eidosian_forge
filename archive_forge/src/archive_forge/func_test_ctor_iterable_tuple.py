import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_ctor_iterable_tuple(self):

    @njit
    def ctor():
        return dict(((1, 2), (1, 2)))
    expected = dict({1: 2})
    got = ctor()
    self.assertEqual(expected, got)