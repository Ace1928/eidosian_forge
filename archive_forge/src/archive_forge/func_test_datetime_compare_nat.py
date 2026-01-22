import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_compare_nat(self):
    dt_nat = np.datetime64('NaT', 'D')
    dt_other = np.datetime64('2000-01-01')
    td_nat = np.timedelta64('NaT', 'h')
    td_other = np.timedelta64(1, 'h')
    for op in [np.equal, np.less, np.less_equal, np.greater, np.greater_equal]:
        assert_(not op(dt_nat, dt_nat))
        assert_(not op(dt_nat, dt_other))
        assert_(not op(dt_other, dt_nat))
        assert_(not op(td_nat, td_nat))
        assert_(not op(td_nat, td_other))
        assert_(not op(td_other, td_nat))
    assert_(np.not_equal(dt_nat, dt_nat))
    assert_(np.not_equal(dt_nat, dt_other))
    assert_(np.not_equal(dt_other, dt_nat))
    assert_(np.not_equal(td_nat, td_nat))
    assert_(np.not_equal(td_nat, td_other))
    assert_(np.not_equal(td_other, td_nat))