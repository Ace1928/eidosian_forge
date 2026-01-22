import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_interpolate_non_ts(self):
    s = Series([1, 3, np.nan, np.nan, np.nan, 11])
    msg = 'time-weighted interpolation only works on Series or DataFrames with a DatetimeIndex'
    with pytest.raises(ValueError, match=msg):
        s.interpolate(method='time')