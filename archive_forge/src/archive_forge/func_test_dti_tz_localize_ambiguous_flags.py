from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_flags(self, tz, unit):
    dr = date_range(datetime(2011, 11, 6, 0), periods=5, freq=offsets.Hour(), tz=tz, unit=unit)
    times = ['11/06/2011 00:00', '11/06/2011 01:00', '11/06/2011 01:00', '11/06/2011 02:00', '11/06/2011 03:00']
    di = DatetimeIndex(times).as_unit(unit)
    is_dst = [1, 1, 0, 0, 0]
    localized = di.tz_localize(tz, ambiguous=is_dst)
    expected = dr._with_freq(None)
    tm.assert_index_equal(expected, localized)
    result = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
    tm.assert_index_equal(result, expected)
    localized = di.tz_localize(tz, ambiguous=np.array(is_dst))
    tm.assert_index_equal(dr, localized)
    localized = di.tz_localize(tz, ambiguous=np.array(is_dst).astype('bool'))
    tm.assert_index_equal(dr, localized)
    localized = DatetimeIndex(times, tz=tz, ambiguous=is_dst).as_unit(unit)
    tm.assert_index_equal(dr, localized)
    times += times
    di = DatetimeIndex(times).as_unit(unit)
    msg = 'Length of ambiguous bool-array must be the same size as vals'
    with pytest.raises(Exception, match=msg):
        di.tz_localize(tz, ambiguous=is_dst)
    is_dst = np.hstack((is_dst, is_dst))
    localized = di.tz_localize(tz, ambiguous=is_dst)
    dr = dr.append(dr)
    tm.assert_index_equal(dr, localized)