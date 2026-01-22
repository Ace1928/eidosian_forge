import datetime
import os
import sys
from os.path import join as pjoin
from io import StringIO
import numpy as np
from numpy.testing import (assert_array_almost_equal,
from pytest import raises as assert_raises
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import read_header, ParseArffError
def test_datetime_local_attribute(self):
    expected = np.array([datetime.datetime(year=1999, month=1, day=31, hour=0, minute=1), datetime.datetime(year=2004, month=12, day=1, hour=23, minute=59), datetime.datetime(year=1817, month=4, day=28, hour=13, minute=0), datetime.datetime(year=2100, month=9, day=10, hour=12, minute=0), datetime.datetime(year=2013, month=11, day=30, hour=4, minute=55), datetime.datetime(year=1631, month=10, day=15, hour=20, minute=4)], dtype='datetime64[m]')
    assert_array_equal(self.data['attr_datetime_local'], expected)