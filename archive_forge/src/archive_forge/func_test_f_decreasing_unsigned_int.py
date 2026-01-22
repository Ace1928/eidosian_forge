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
@pytest.mark.parametrize('f_dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
def test_f_decreasing_unsigned_int(self, f_dtype):
    f = np.array([5, 4, 3, 2, 1], dtype=f_dtype)
    g = gradient(f)
    assert_array_equal(g, [-1] * len(f))