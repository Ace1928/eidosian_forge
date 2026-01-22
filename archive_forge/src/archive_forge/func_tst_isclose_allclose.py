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
def tst_isclose_allclose(self, x, y):
    msg = "isclose.all() and allclose aren't same for %s and %s"
    msg2 = "isclose and allclose aren't same for %s and %s"
    if np.isscalar(x) and np.isscalar(y):
        assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
    else:
        assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))