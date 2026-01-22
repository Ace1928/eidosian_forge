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
def test_logical_not_abs(self):
    assert_array_equal(~self.t, self.f)
    assert_array_equal(np.abs(~self.t), self.f)
    assert_array_equal(np.abs(~self.f), self.t)
    assert_array_equal(np.abs(self.f), self.f)
    assert_array_equal(~np.abs(self.f), self.t)
    assert_array_equal(~np.abs(self.t), self.f)
    assert_array_equal(np.abs(~self.nm), self.im)
    np.logical_not(self.t, out=self.o)
    assert_array_equal(self.o, self.f)
    np.abs(self.t, out=self.o)
    assert_array_equal(self.o, self.t)