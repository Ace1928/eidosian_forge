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
def test_nan_behavior(self):
    a = np.arange(24, dtype=float)
    a[2] = np.nan
    assert_equal(np.median(a), np.nan)
    assert_equal(np.median(a, axis=0), np.nan)
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    a[1, 2, 3] = np.nan
    a[1, 1, 2] = np.nan
    assert_equal(np.median(a), np.nan)
    assert_equal(np.median(a).ndim, 0)
    b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 0)
    b[2, 3] = np.nan
    b[1, 2] = np.nan
    assert_equal(np.median(a, 0), b)
    b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), 1)
    b[1, 3] = np.nan
    b[1, 2] = np.nan
    assert_equal(np.median(a, 1), b)
    b = np.median(np.arange(24, dtype=float).reshape(2, 3, 4), (0, 2))
    b[1] = np.nan
    b[2] = np.nan
    assert_equal(np.median(a, (0, 2)), b)