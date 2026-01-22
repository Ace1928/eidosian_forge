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
def test_literal_nested(self):

    @njit
    def foo(x):
        return literally(x) * 2

    @njit
    def bar(y, x):
        return foo(y) + x
    y, x = (3, 7)
    self.assertEqual(bar(y, x), y * 2 + x)
    [foo_sig] = foo.signatures
    self.assertEqual(foo_sig[0], types.literal(y))
    [bar_sig] = bar.signatures
    self.assertEqual(bar_sig[0], types.literal(y))
    self.assertNotIsInstance(bar_sig[1], types.Literal)