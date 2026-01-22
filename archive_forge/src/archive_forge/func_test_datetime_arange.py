import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_arange(self):
    a = np.arange('2010-01-05', '2010-01-10', dtype='M8[D]')
    assert_equal(a.dtype, np.dtype('M8[D]'))
    assert_equal(a, np.array(['2010-01-05', '2010-01-06', '2010-01-07', '2010-01-08', '2010-01-09'], dtype='M8[D]'))
    a = np.arange('1950-02-10', '1950-02-06', -1, dtype='M8[D]')
    assert_equal(a.dtype, np.dtype('M8[D]'))
    assert_equal(a, np.array(['1950-02-10', '1950-02-09', '1950-02-08', '1950-02-07'], dtype='M8[D]'))
    a = np.arange('1969-05', '1970-05', 2, dtype='M8')
    assert_equal(a.dtype, np.dtype('M8[M]'))
    assert_equal(a, np.datetime64('1969-05') + np.arange(12, step=2))
    a = np.arange('1969', 18, 3, dtype='M8')
    assert_equal(a.dtype, np.dtype('M8[Y]'))
    assert_equal(a, np.datetime64('1969') + np.arange(18, step=3))
    a = np.arange('1969-12-19', 22, np.timedelta64(2), dtype='M8')
    assert_equal(a.dtype, np.dtype('M8[D]'))
    assert_equal(a, np.datetime64('1969-12-19') + np.arange(22, step=2))
    assert_raises(ValueError, np.arange, np.datetime64('today'), np.datetime64('today') + 3, 0)
    assert_raises(TypeError, np.arange, np.datetime64('2011-03-01', 'D'), np.timedelta64(5, 'M'))
    assert_raises(TypeError, np.arange, np.datetime64('2012-02-03T14', 's'), np.timedelta64(5, 'Y'))