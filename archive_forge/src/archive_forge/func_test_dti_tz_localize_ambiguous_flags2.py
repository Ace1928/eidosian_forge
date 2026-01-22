from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_flags2(self, tz, unit):
    dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
    is_dst = np.array([1] * 10)
    localized = dr.tz_localize(tz)
    localized_is_dst = dr.tz_localize(tz, ambiguous=is_dst)
    tm.assert_index_equal(localized, localized_is_dst)