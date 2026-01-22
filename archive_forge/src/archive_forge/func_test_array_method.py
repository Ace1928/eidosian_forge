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
def test_array_method(self):
    m = np.array([[1, 0, 0], [4, 0, 6]])
    tgt = [[0, 1, 1], [0, 0, 2]]
    assert_equal(m.nonzero(), tgt)