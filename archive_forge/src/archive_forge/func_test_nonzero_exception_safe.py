import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_nonzero_exception_safe(self):

    class ThrowsAfter:

        def __init__(self, iters):
            self.iters_left = iters

        def __bool__(self):
            if self.iters_left == 0:
                raise ValueError('called `iters` times')
            self.iters_left -= 1
            return True
    '\n        Test that a ValueError is raised instead of a SystemError\n\n        If the __bool__ function is called after the error state is set,\n        Python (cpython) will raise a SystemError.\n        '
    a = np.array([ThrowsAfter(5)] * 10)
    assert_raises(ValueError, np.nonzero, a)
    a = np.array([ThrowsAfter(15)] * 10)
    assert_raises(ValueError, np.nonzero, a)
    a = np.array([[ThrowsAfter(15)]] * 10)
    assert_raises(ValueError, np.nonzero, a)