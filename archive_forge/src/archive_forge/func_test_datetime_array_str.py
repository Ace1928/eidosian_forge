import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_array_str(self):
    a = np.array(['2011-03-16', '1920-01-01', '2013-05-19'], dtype='M')
    assert_equal(str(a), "['2011-03-16' '1920-01-01' '2013-05-19']")
    a = np.array(['2011-03-16T13:55', '1920-01-01T03:12'], dtype='M')
    assert_equal(np.array2string(a, separator=', ', formatter={'datetime': lambda x: "'%s'" % np.datetime_as_string(x, timezone='UTC')}), "['2011-03-16T13:55Z', '1920-01-01T03:12Z']")
    a = np.array(['2010', 'NaT', '2030']).astype('M')
    assert_equal(str(a), "['2010'  'NaT' '2030']")