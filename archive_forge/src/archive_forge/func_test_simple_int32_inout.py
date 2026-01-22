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
@pytest.mark.parametrize('casting', [None, 'unsafe'])
def test_simple_int32_inout(self, casting):
    a = self._generate_int32_data(self.nr, self.nc)
    m = np.float64(0)
    M = np.float64(2)
    ac = np.zeros(a.shape, dtype=np.int32)
    act = ac.copy()
    if casting is None:
        with pytest.raises(TypeError):
            self.fastclip(a, m, M, ac, casting=casting)
    else:
        self.fastclip(a, m, M, ac, casting=casting)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)