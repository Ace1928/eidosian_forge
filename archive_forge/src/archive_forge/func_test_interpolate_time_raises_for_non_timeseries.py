import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_time_raises_for_non_timeseries(self):
    non_ts = Series([0, 1, 2, np.nan])
    msg = 'time-weighted interpolation only works on Series.* with a DatetimeIndex'
    with pytest.raises(ValueError, match=msg):
        non_ts.interpolate(method='time')