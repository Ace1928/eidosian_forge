import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_timedelta_array_str(self):
    a = np.array([-1, 0, 100], dtype='m')
    assert_equal(str(a), '[ -1   0 100]')
    a = np.array(['NaT', 'NaT'], dtype='m')
    assert_equal(str(a), "['NaT' 'NaT']")
    a = np.array([-1, 'NaT', 0], dtype='m')
    assert_equal(str(a), "[   -1 'NaT'     0]")
    a = np.array([-1, 'NaT', 1234567], dtype='m')
    assert_equal(str(a), "[     -1   'NaT' 1234567]")
    a = np.array([-1, 'NaT', 1234567], dtype='>m')
    assert_equal(str(a), "[     -1   'NaT' 1234567]")
    a = np.array([-1, 'NaT', 1234567], dtype='<m')
    assert_equal(str(a), "[     -1   'NaT' 1234567]")