import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.parametrize('op1, op2', [(np.timedelta64(7, 's'), np.timedelta64(4, 's')), (np.timedelta64(7, 's'), np.timedelta64(-4, 's')), (np.timedelta64(8, 's'), np.timedelta64(-4, 's')), (np.timedelta64(1, 'm'), np.timedelta64(31, 's')), (np.timedelta64(1890), np.timedelta64(31)), (np.timedelta64(2, 'Y'), np.timedelta64('13', 'M')), (np.array([1, 2, 3], dtype='m8'), np.array([2], dtype='m8'))])
def test_timedelta_divmod(self, op1, op2):
    expected = (op1 // op2, op1 % op2)
    assert_equal(divmod(op1, op2), expected)