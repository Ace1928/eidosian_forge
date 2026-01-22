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
def test_neg_width_boundaries(self):
    assert_equal(np.binary_repr(-128, width=8), '10000000')
    for width in range(1, 11):
        num = -2 ** (width - 1)
        exp = '1' + (width - 1) * '0'
        assert_equal(np.binary_repr(num, width=width), exp)