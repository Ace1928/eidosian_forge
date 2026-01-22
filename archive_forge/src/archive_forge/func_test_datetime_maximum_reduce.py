import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_maximum_reduce(self):
    a = np.array(['2010-01-02', '1999-03-14', '1833-03'], dtype='M8[D]')
    assert_equal(np.maximum.reduce(a).dtype, np.dtype('M8[D]'))
    assert_equal(np.maximum.reduce(a), np.datetime64('2010-01-02'))
    a = np.array([1, 4, 0, 7, 2], dtype='m8[s]')
    assert_equal(np.maximum.reduce(a).dtype, np.dtype('m8[s]'))
    assert_equal(np.maximum.reduce(a), np.timedelta64(7, 's'))