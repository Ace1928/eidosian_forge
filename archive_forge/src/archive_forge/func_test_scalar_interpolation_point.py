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
def test_scalar_interpolation_point(self):
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    x0 = 0
    assert_almost_equal(np.interp(x0, x, y), x0)
    x0 = 0.3
    assert_almost_equal(np.interp(x0, x, y), x0)
    x0 = np.float32(0.3)
    assert_almost_equal(np.interp(x0, x, y), x0)
    x0 = np.float64(0.3)
    assert_almost_equal(np.interp(x0, x, y), x0)
    x0 = np.nan
    assert_almost_equal(np.interp(x0, x, y), x0)