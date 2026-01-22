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
@pytest.mark.xfail(reason="propagation doesn't match spec")
@pytest.mark.parametrize('arr, amin, amax', [(np.array([1] * 10, dtype='m8'), np.timedelta64('NaT'), np.zeros(10, dtype=np.int32))])
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_NaT_propagation(self, arr, amin, amax):
    expected = np.minimum(np.maximum(arr, amin), amax)
    actual = np.clip(arr, amin, amax)
    assert_equal(actual, expected)