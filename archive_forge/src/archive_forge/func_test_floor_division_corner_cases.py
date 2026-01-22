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
@pytest.mark.parametrize('dtype', np.typecodes['Float'])
def test_floor_division_corner_cases(self, dtype):
    x = np.zeros(10, dtype=dtype)
    y = np.ones(10, dtype=dtype)
    fnan = np.array(np.nan, dtype=dtype)
    fone = np.array(1.0, dtype=dtype)
    fzer = np.array(0.0, dtype=dtype)
    finf = np.array(np.inf, dtype=dtype)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in floor_divide')
        div = np.floor_divide(fnan, fone)
        assert np.isnan(div), 'dt: %s, div: %s' % (dt, div)
        div = np.floor_divide(fone, fnan)
        assert np.isnan(div), 'dt: %s, div: %s' % (dt, div)
        div = np.floor_divide(fnan, fzer)
        assert np.isnan(div), 'dt: %s, div: %s' % (dt, div)
    with np.errstate(divide='ignore'):
        z = np.floor_divide(y, x)
        assert_(np.isinf(z).all())