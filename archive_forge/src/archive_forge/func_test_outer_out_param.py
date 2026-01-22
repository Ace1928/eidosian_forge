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
def test_outer_out_param():
    arr1 = np.ones((5,))
    arr2 = np.ones((2,))
    arr3 = np.linspace(-2, 2, 5)
    out1 = np.ndarray(shape=(5, 5))
    out2 = np.ndarray(shape=(2, 5))
    res1 = np.outer(arr1, arr3, out1)
    assert_equal(res1, out1)
    assert_equal(np.outer(arr2, arr3, out2), out2)