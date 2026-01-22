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
def test_simple_int(self):
    a = self._generate_int_data(self.nr, self.nc)
    a = a.astype(int)
    m = -2
    M = 4
    ac = self.fastclip(a, m, M)
    act = self.clip(a, m, M)
    assert_array_strict_equal(ac, act)