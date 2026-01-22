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
def test_type_cast_04(self):
    a = self._generate_int32_data(self.nr, self.nc)
    m = np.float32(-2)
    M = np.float32(4)
    act = self.fastclip(a, m, M)
    ac = self.clip(a, m, M)
    assert_array_strict_equal(ac, act)