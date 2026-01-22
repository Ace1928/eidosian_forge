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
def test_no_parameter_modification(self):
    x = np.array([np.inf, 1])
    y = np.array([0, np.inf])
    np.isclose(x, y)
    assert_array_equal(x, np.array([np.inf, 1]))
    assert_array_equal(y, np.array([0, np.inf]))