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
def test_base_range(self):
    with assert_raises(ValueError):
        np.base_repr(1, 1)
    with assert_raises(ValueError):
        np.base_repr(1, 37)