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
@pytest.mark.parametrize('arr, amin, amax, exp', [(np.zeros(10, dtype=np.int64), 0, -2 ** 64 + 1, np.full(10, -2 ** 64 + 1, dtype=object)), (np.zeros(10, dtype='m8') - 1, 0, 0, np.zeros(10, dtype='m8'))])
def test_clip_problem_cases(self, arr, amin, amax, exp):
    actual = np.clip(arr, amin, amax)
    assert_equal(actual, exp)