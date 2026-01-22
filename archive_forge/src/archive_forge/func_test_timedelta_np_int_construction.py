import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('unit', ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as', 'generic'])
def test_timedelta_np_int_construction(self, unit):
    if unit != 'generic':
        assert_equal(np.timedelta64(np.int64(123), unit), np.timedelta64(123, unit))
    else:
        assert_equal(np.timedelta64(np.int64(123)), np.timedelta64(123))