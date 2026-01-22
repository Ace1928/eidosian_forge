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
def test_type_cast_12(self):
    a = self._generate_int_data(self.nr, self.nc)
    b = np.zeros(a.shape, dtype=np.float32)
    m = np.int32(0)
    M = np.int32(1)
    act = self.clip(a, m, M, out=b)
    ac = self.fastclip(a, m, M, out=b)
    assert_array_strict_equal(ac, act)