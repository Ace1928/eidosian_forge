import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_keywords2_ticket_2100(self):

    def foo(a, b=1):
        return a + b
    f = vectorize(foo)
    args = np.array([1, 2, 3])
    r1 = f(a=args)
    r2 = np.array([2, 3, 4])
    assert_array_equal(r1, r2)
    r1 = f(b=1, a=args)
    assert_array_equal(r1, r2)
    r1 = f(args, b=2)
    r2 = np.array([3, 4, 5])
    assert_array_equal(r1, r2)