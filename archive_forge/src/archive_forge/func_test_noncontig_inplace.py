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
def test_noncontig_inplace(self):
    a = self._generate_data(self.nr * 2, self.nc * 3)
    a = a[::2, ::3]
    assert_(not a.flags['F_CONTIGUOUS'])
    assert_(not a.flags['C_CONTIGUOUS'])
    ac = a.copy()
    m = -0.5
    M = 0.6
    self.fastclip(a, m, M, a)
    self.clip(ac, m, M, ac)
    assert_array_equal(a, ac)