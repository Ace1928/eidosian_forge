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
def test_masked_arrays(self):
    x = np.ma.masked_where([True, True, False], np.arange(3))
    assert_(type(x) is type(np.isclose(2, x)))
    assert_(type(x) is type(np.isclose(x, 2)))
    x = np.ma.masked_where([True, True, False], [np.nan, np.inf, np.nan])
    assert_(type(x) is type(np.isclose(np.inf, x)))
    assert_(type(x) is type(np.isclose(x, np.inf)))
    x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
    y = np.isclose(np.nan, x, equal_nan=True)
    assert_(type(x) is type(y))
    assert_array_equal([True, True, False], y.mask)
    y = np.isclose(x, np.nan, equal_nan=True)
    assert_(type(x) is type(y))
    assert_array_equal([True, True, False], y.mask)
    x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
    y = np.isclose(x, x, equal_nan=True)
    assert_(type(x) is type(y))
    assert_array_equal([True, True, False], y.mask)