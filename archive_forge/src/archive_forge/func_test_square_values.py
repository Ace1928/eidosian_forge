import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
def test_square_values(self):
    x = [np.nan, np.nan, np.inf, np.inf]
    y = [np.nan, -np.nan, np.inf, -np.inf]
    with np.errstate(all='ignore'):
        for dt in ['e', 'f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.square(yf), xf)
    with np.errstate(over='raise'):
        assert_raises(FloatingPointError, np.square, np.array(1000.0, dtype='e'))
        assert_raises(FloatingPointError, np.square, np.array(1e+32, dtype='f'))
        assert_raises(FloatingPointError, np.square, np.array(1e+200, dtype='d'))