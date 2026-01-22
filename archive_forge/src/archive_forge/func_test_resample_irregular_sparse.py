from datetime import datetime
import warnings
import dateutil
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import _get_period_range_edges
from pandas.tseries import offsets
def test_resample_irregular_sparse(self):
    dr = date_range(start='1/1/2012', freq='5min', periods=1000)
    s = Series(np.array(100), index=dr)
    subset = s[:'2012-01-04 06:55']
    result = subset.resample('10min').apply(len)
    expected = s.resample('10min').apply(len).loc[result.index]
    tm.assert_series_equal(result, expected)