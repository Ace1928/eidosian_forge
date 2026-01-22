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
@pytest.mark.parametrize('x_dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
def test_x_decreasing_unsigned(self, x_dtype):
    x = np.array([3, 2, 1], dtype=x_dtype)
    f = np.array([0, 2, 4])
    dfdx = gradient(f, x)
    assert_array_equal(dfdx, [-2] * len(x))