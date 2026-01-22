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
def test_month_attribute(self):
    expected = np.array(['1999-01', '2004-12', '1817-04', '2100-09', '2013-11', '1631-10'], dtype='datetime64[M]')
    assert_array_equal(self.data['attr_month'], expected)