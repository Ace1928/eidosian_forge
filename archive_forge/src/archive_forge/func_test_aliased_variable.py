import numpy as np
import numba
import unittest
from numba.tests.support import TestCase
from numba import njit
from numba.core import types, errors, cgutils
from numba.core.typing import signature
from numba.core.datamodel import models
from numba.core.extending import (
from numba.misc.special import literally
def test_aliased_variable(self):

    @njit
    def foo(a, b, c):

        def closure(d):
            return literally(d) + 10 * inner(a, b)
        return closure(c)

    @njit
    def inner(x, y):
        return x + literally(y)
    kwargs = dict(a=1, b=2, c=3)
    got = foo(**kwargs)
    expect = (lambda a, b, c: c + 10 * (a + b))(**kwargs)
    self.assertEqual(got, expect)
    [(type_a, type_b, type_c)] = foo.signatures
    self.assertNotIsInstance(type_a, types.Literal)
    self.assertIsInstance(type_b, types.Literal)
    self.assertEqual(type_b.literal_value, 2)
    self.assertIsInstance(type_c, types.Literal)
    self.assertEqual(type_c.literal_value, 3)