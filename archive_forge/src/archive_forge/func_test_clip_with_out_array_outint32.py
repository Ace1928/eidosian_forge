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
def test_clip_with_out_array_outint32(self):
    a = self._generate_data(self.nr, self.nc)
    m = -1.0
    M = 2.0
    ac = np.zeros(a.shape, dtype=np.int32)
    act = ac.copy()
    self.fastclip(a, m, M, out=ac, casting='unsafe')
    self.clip(a, m, M, act)
    assert_array_strict_equal(ac, act)