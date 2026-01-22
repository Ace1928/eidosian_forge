import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_in1d_table_timedelta_fails(self):
    a = np.array([0, 1, 2], dtype='timedelta64[s]')
    b = a
    with pytest.raises(ValueError):
        in1d(a, b, kind='table')