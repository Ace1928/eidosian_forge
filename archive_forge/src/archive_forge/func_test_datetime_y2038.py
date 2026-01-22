import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_y2038(self):
    a = np.datetime64('2038-01-19T03:14:07')
    assert_equal(a.view(np.int64), 2 ** 31 - 1)
    a = np.datetime64('2038-01-19T03:14:08')
    assert_equal(a.view(np.int64), 2 ** 31)
    with assert_warns(DeprecationWarning):
        a = np.datetime64('2038-01-19T04:14:07+0100')
        assert_equal(a.view(np.int64), 2 ** 31 - 1)
    with assert_warns(DeprecationWarning):
        a = np.datetime64('2038-01-19T04:14:08+0100')
        assert_equal(a.view(np.int64), 2 ** 31)
    a = np.datetime64('2038-01-20T13:21:14')
    assert_equal(str(a), '2038-01-20T13:21:14')