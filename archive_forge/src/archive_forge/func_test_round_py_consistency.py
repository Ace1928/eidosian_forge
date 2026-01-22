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
@pytest.mark.xfail(raises=AssertionError, reason='gh-15896')
def test_round_py_consistency(self):
    f = 5.1 * 10 ** 73
    assert_equal(round(np.float64(f), -73), round(f, -73))