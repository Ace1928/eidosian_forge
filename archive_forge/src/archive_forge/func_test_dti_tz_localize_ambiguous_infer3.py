from datetime import (
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', easts)
def test_dti_tz_localize_ambiguous_infer3(self, tz):
    dr = date_range(datetime(2011, 6, 1, 0), periods=10, freq=offsets.Hour())
    localized = dr.tz_localize(tz)
    localized_infer = dr.tz_localize(tz, ambiguous='infer')
    tm.assert_index_equal(localized, localized_infer)