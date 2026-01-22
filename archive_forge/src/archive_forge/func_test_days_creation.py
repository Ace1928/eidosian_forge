import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_days_creation(self):
    assert_equal(np.array('1599', dtype='M8[D]').astype('i8'), (1600 - 1970) * 365 - (1972 - 1600) / 4 + 3 - 365)
    assert_equal(np.array('1600', dtype='M8[D]').astype('i8'), (1600 - 1970) * 365 - (1972 - 1600) / 4 + 3)
    assert_equal(np.array('1601', dtype='M8[D]').astype('i8'), (1600 - 1970) * 365 - (1972 - 1600) / 4 + 3 + 366)
    assert_equal(np.array('1900', dtype='M8[D]').astype('i8'), (1900 - 1970) * 365 - (1970 - 1900) // 4)
    assert_equal(np.array('1901', dtype='M8[D]').astype('i8'), (1900 - 1970) * 365 - (1970 - 1900) // 4 + 365)
    assert_equal(np.array('1967', dtype='M8[D]').astype('i8'), -3 * 365 - 1)
    assert_equal(np.array('1968', dtype='M8[D]').astype('i8'), -2 * 365 - 1)
    assert_equal(np.array('1969', dtype='M8[D]').astype('i8'), -1 * 365)
    assert_equal(np.array('1970', dtype='M8[D]').astype('i8'), 0 * 365)
    assert_equal(np.array('1971', dtype='M8[D]').astype('i8'), 1 * 365)
    assert_equal(np.array('1972', dtype='M8[D]').astype('i8'), 2 * 365)
    assert_equal(np.array('1973', dtype='M8[D]').astype('i8'), 3 * 365 + 1)
    assert_equal(np.array('1974', dtype='M8[D]').astype('i8'), 4 * 365 + 1)
    assert_equal(np.array('2000', dtype='M8[D]').astype('i8'), (2000 - 1970) * 365 + (2000 - 1972) // 4)
    assert_equal(np.array('2001', dtype='M8[D]').astype('i8'), (2000 - 1970) * 365 + (2000 - 1972) // 4 + 366)
    assert_equal(np.array('2400', dtype='M8[D]').astype('i8'), (2400 - 1970) * 365 + (2400 - 1972) // 4 - 3)
    assert_equal(np.array('2401', dtype='M8[D]').astype('i8'), (2400 - 1970) * 365 + (2400 - 1972) // 4 - 3 + 366)
    assert_equal(np.array('1600-02-29', dtype='M8[D]').astype('i8'), (1600 - 1970) * 365 - (1972 - 1600) // 4 + 3 + 31 + 28)
    assert_equal(np.array('1600-03-01', dtype='M8[D]').astype('i8'), (1600 - 1970) * 365 - (1972 - 1600) // 4 + 3 + 31 + 29)
    assert_equal(np.array('2000-02-29', dtype='M8[D]').astype('i8'), (2000 - 1970) * 365 + (2000 - 1972) // 4 + 31 + 28)
    assert_equal(np.array('2000-03-01', dtype='M8[D]').astype('i8'), (2000 - 1970) * 365 + (2000 - 1972) // 4 + 31 + 29)
    assert_equal(np.array('2001-03-22', dtype='M8[D]').astype('i8'), (2000 - 1970) * 365 + (2000 - 1972) // 4 + 366 + 31 + 28 + 21)