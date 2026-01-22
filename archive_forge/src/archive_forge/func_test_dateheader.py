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
def test_dateheader(self):
    with open(test7) as ofile:
        rel, attrs = read_header(ofile)
    assert_(rel == 'test7')
    assert_(len(attrs) == 5)
    assert_(attrs[0].name == 'attr_year')
    assert_(attrs[0].date_format == '%Y')
    assert_(attrs[1].name == 'attr_month')
    assert_(attrs[1].date_format == '%Y-%m')
    assert_(attrs[2].name == 'attr_date')
    assert_(attrs[2].date_format == '%Y-%m-%d')
    assert_(attrs[3].name == 'attr_datetime_local')
    assert_(attrs[3].date_format == '%Y-%m-%d %H:%M')
    assert_(attrs[4].name == 'attr_datetime_missing')
    assert_(attrs[4].date_format == '%Y-%m-%d %H:%M')