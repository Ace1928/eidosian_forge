import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_arange_no_dtype(self):
    d = np.array('2010-01-04', dtype='M8[D]')
    assert_equal(np.arange(d, d + 1), d)
    assert_raises(ValueError, np.arange, d)