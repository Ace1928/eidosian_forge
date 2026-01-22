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
def test_logaddexp2_values(self):
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    z = [6, 6, 6, 6, 6]
    for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
        xf = np.log2(np.array(x, dtype=dt))
        yf = np.log2(np.array(y, dtype=dt))
        zf = np.log2(np.array(z, dtype=dt))
        assert_almost_equal(np.logaddexp2(xf, yf), zf, decimal=dec_)