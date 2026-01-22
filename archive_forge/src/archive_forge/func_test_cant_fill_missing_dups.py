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
def test_cant_fill_missing_dups(self):
    rng = PeriodIndex([2000, 2005, 2005, 2007, 2007], freq='Y')
    s = Series(np.random.default_rng(2).standard_normal(5), index=rng)
    msg = 'Reindexing only valid with uniquely valued Index objects'
    with pytest.raises(InvalidIndexError, match=msg):
        s.resample('Y').ffill()