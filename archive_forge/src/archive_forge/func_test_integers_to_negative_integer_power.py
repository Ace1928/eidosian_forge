import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
def test_integers_to_negative_integer_power(self):
    exp = [np.array(-1, dt)[()] for dt in 'bhilq']
    base = [np.array(1, dt)[()] for dt in 'bhilqBHILQ']
    for i1, i2 in itertools.product(base, exp):
        if i1.dtype != np.uint64:
            assert_raises(ValueError, operator.pow, i1, i2)
        else:
            res = operator.pow(i1, i2)
            assert_(res.dtype.type is np.float64)
            assert_almost_equal(res, 1.0)
    base = [np.array(-1, dt)[()] for dt in 'bhilq']
    for i1, i2 in itertools.product(base, exp):
        if i1.dtype != np.uint64:
            assert_raises(ValueError, operator.pow, i1, i2)
        else:
            res = operator.pow(i1, i2)
            assert_(res.dtype.type is np.float64)
            assert_almost_equal(res, -1.0)
    base = [np.array(2, dt)[()] for dt in 'bhilqBHILQ']
    for i1, i2 in itertools.product(base, exp):
        if i1.dtype != np.uint64:
            assert_raises(ValueError, operator.pow, i1, i2)
        else:
            res = operator.pow(i1, i2)
            assert_(res.dtype.type is np.float64)
            assert_almost_equal(res, 0.5)