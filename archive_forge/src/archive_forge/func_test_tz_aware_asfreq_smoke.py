from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
def test_tz_aware_asfreq_smoke(self, tz, frame_or_series):
    dr = date_range('2011-12-01', '2012-07-20', freq='D', tz=tz)
    obj = frame_or_series(np.random.default_rng(2).standard_normal(len(dr)), index=dr)
    obj.asfreq('min')