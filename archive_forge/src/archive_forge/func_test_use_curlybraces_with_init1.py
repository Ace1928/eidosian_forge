import numpy as np
from numba import njit, jit
from numba.core.errors import TypingError
import unittest
from numba.tests.support import TestCase
def test_use_curlybraces_with_init1(self):

    @njit
    def foo():
        return {1: 2}
    d = foo()
    self.assertEqual(d, {1: 2})