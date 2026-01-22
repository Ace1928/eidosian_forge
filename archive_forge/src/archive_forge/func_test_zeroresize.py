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
def test_zeroresize(self):
    A = np.array([[1, 2], [3, 4]])
    Ar = np.resize(A, (0,))
    assert_array_equal(Ar, np.array([]))
    assert_equal(A.dtype, Ar.dtype)
    Ar = np.resize(A, (0, 2))
    assert_equal(Ar.shape, (0, 2))
    Ar = np.resize(A, (2, 0))
    assert_equal(Ar.shape, (2, 0))