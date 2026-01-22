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
def test_none_compares_elementwise(self):
    a = np.array([None, 1, None], dtype=object)
    assert_equal(a == None, [True, False, True])
    assert_equal(a != None, [False, True, False])
    a = np.ones(3)
    assert_equal(a == None, [False, False, False])
    assert_equal(a != None, [True, True, True])