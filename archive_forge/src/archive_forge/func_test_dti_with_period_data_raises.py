from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_dti_with_period_data_raises(self):
    data = pd.PeriodIndex(['2016Q1', '2016Q2'], freq='Q')
    with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
        DatetimeIndex(data)
    with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
        to_datetime(data)
    with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
        DatetimeIndex(period_array(data))
    with pytest.raises(TypeError, match='PeriodDtype data is invalid'):
        to_datetime(period_array(data))