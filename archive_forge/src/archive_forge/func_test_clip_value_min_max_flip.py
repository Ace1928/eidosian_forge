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
@pytest.mark.parametrize('amin, amax', [(1, 0), (1, np.zeros(10)), (np.ones(10), np.zeros(10))])
def test_clip_value_min_max_flip(self, amin, amax):
    a = np.arange(10, dtype=np.int64)
    expected = np.minimum(np.maximum(a, amin), amax)
    actual = np.clip(a, amin, amax)
    assert_equal(actual, expected)