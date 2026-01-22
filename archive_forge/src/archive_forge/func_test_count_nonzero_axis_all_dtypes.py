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
def test_count_nonzero_axis_all_dtypes(self):
    msg = 'Mismatch for dtype: %s'

    def assert_equal_w_dt(a, b, err_msg):
        assert_equal(a.dtype, b.dtype, err_msg=err_msg)
        assert_equal(a, b, err_msg=err_msg)
    for dt in np.typecodes['All']:
        err_msg = msg % (np.dtype(dt).name,)
        if dt != 'V':
            if dt != 'M':
                m = np.zeros((3, 3), dtype=dt)
                n = np.ones(1, dtype=dt)
                m[0, 0] = n[0]
                m[1, 0] = n[0]
            else:
                m = np.array(['1970-01-01'] * 9)
                m = m.reshape((3, 3))
                m[0, 0] = '1970-01-12'
                m[1, 0] = '1970-01-12'
                m = m.astype(dt)
            expected = np.array([2, 0, 0], dtype=np.intp)
            assert_equal_w_dt(np.count_nonzero(m, axis=0), expected, err_msg=err_msg)
            expected = np.array([1, 1, 0], dtype=np.intp)
            assert_equal_w_dt(np.count_nonzero(m, axis=1), expected, err_msg=err_msg)
            expected = np.array(2)
            assert_equal(np.count_nonzero(m, axis=(0, 1)), expected, err_msg=err_msg)
            assert_equal(np.count_nonzero(m, axis=None), expected, err_msg=err_msg)
            assert_equal(np.count_nonzero(m), expected, err_msg=err_msg)
        if dt == 'V':
            m = np.array([np.void(1)] * 6).reshape((2, 3))
            expected = np.array([0, 0, 0], dtype=np.intp)
            assert_equal_w_dt(np.count_nonzero(m, axis=0), expected, err_msg=err_msg)
            expected = np.array([0, 0], dtype=np.intp)
            assert_equal_w_dt(np.count_nonzero(m, axis=1), expected, err_msg=err_msg)
            expected = np.array(0)
            assert_equal(np.count_nonzero(m, axis=(0, 1)), expected, err_msg=err_msg)
            assert_equal(np.count_nonzero(m, axis=None), expected, err_msg=err_msg)
            assert_equal(np.count_nonzero(m), expected, err_msg=err_msg)