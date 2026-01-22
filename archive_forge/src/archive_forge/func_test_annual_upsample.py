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
def test_annual_upsample(self, simple_period_range_series):
    ts = simple_period_range_series('1/1/1990', '12/31/1995', freq='Y-DEC')
    df = DataFrame({'a': ts})
    rdf = df.resample('D').ffill()
    exp = df['a'].resample('D').ffill()
    tm.assert_series_equal(rdf['a'], exp)