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
def test_object_clip(self):
    a = np.arange(10, dtype=object)
    actual = np.clip(a, 1, 5)
    expected = np.array([1, 1, 2, 3, 4, 5, 5, 5, 5, 5])
    assert actual.tolist() == expected.tolist()