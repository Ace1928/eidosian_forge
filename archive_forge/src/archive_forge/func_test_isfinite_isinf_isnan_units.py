import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('unit', ['Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'])
@pytest.mark.parametrize('dstr', ['<datetime64[%s]', '>datetime64[%s]', '<timedelta64[%s]', '>timedelta64[%s]'])
def test_isfinite_isinf_isnan_units(self, unit, dstr):
    """check isfinite, isinf, isnan for all units of <M, >M, <m, >m dtypes
        """
    arr_val = [123, -321, 'NaT']
    arr = np.array(arr_val, dtype=dstr % unit)
    pos = np.array([True, True, False])
    neg = np.array([False, False, True])
    false = np.array([False, False, False])
    assert_equal(np.isfinite(arr), pos)
    assert_equal(np.isinf(arr), false)
    assert_equal(np.isnan(arr), neg)