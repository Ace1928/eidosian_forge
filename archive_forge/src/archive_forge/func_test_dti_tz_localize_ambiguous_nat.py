from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_nat(self, tz):
    times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
    di = DatetimeIndex(times)
    localized = di.tz_localize(tz, ambiguous='NaT')
    times = ['11/06/2011 00:00', np.nan, np.nan, '11/06/2011 02:00', '11/06/2011 03:00']
    di_test = DatetimeIndex(times, tz='US/Eastern')
    tm.assert_numpy_array_equal(di_test.values, localized.values)