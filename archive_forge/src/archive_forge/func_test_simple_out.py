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
def test_simple_out(self):
    a = self._generate_data(self.nr, self.nc)
    m = -0.5
    M = 0.6
    ac = np.zeros(a.shape)
    act = np.zeros(a.shape)
    self.fastclip(a, m, M, ac)
    self.clip(a, m, M, act)
    assert_array_strict_equal(ac, act)