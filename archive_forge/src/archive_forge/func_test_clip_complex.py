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
def test_clip_complex(self):
    a = np.ones(10, dtype=complex)
    m = a.min()
    M = a.max()
    am = self.fastclip(a, m, None)
    aM = self.fastclip(a, None, M)
    assert_array_strict_equal(am, a)
    assert_array_strict_equal(aM, a)