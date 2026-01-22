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
def test_datetime_missing(self):
    expected = np.array(['nat', '2004-12-01T23:59', 'nat', 'nat', '2013-11-30T04:55', '1631-10-15T20:04'], dtype='datetime64[m]')
    assert_array_equal(self.data['attr_datetime_missing'], expected)