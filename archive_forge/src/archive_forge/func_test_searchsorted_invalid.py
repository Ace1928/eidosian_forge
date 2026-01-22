import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_searchsorted_invalid(self):
    pidx = PeriodIndex(['2014-01-01', '2014-01-02', '2014-01-03', '2014-01-04', '2014-01-05'], freq='D')
    other = np.array([0, 1], dtype=np.int64)
    msg = '|'.join(['searchsorted requires compatible dtype or scalar', "value should be a 'Period', 'NaT', or array of those. Got"])
    with pytest.raises(TypeError, match=msg):
        pidx.searchsorted(other)
    with pytest.raises(TypeError, match=msg):
        pidx.searchsorted(other.astype('timedelta64[ns]'))
    with pytest.raises(TypeError, match=msg):
        pidx.searchsorted(np.timedelta64(4))
    with pytest.raises(TypeError, match=msg):
        pidx.searchsorted(np.timedelta64('NaT', 'ms'))
    with pytest.raises(TypeError, match=msg):
        pidx.searchsorted(np.datetime64(4, 'ns'))
    with pytest.raises(TypeError, match=msg):
        pidx.searchsorted(np.datetime64('NaT', 'ns'))