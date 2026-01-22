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
def test_returned(self):
    y = np.array([[1, 2, 3], [4, 5, 6]])
    avg, scl = average(y, returned=True)
    assert_equal(scl, 6.0)
    avg, scl = average(y, 0, returned=True)
    assert_array_equal(scl, np.array([2.0, 2.0, 2.0]))
    avg, scl = average(y, 1, returned=True)
    assert_array_equal(scl, np.array([3.0, 3.0]))
    w0 = [1, 2]
    avg, scl = average(y, weights=w0, axis=0, returned=True)
    assert_array_equal(scl, np.array([3.0, 3.0, 3.0]))
    w1 = [1, 2, 3]
    avg, scl = average(y, weights=w1, axis=1, returned=True)
    assert_array_equal(scl, np.array([6.0, 6.0]))
    w2 = [[0, 0, 1], [1, 2, 3]]
    avg, scl = average(y, weights=w2, axis=1, returned=True)
    assert_array_equal(scl, np.array([1.0, 6.0]))