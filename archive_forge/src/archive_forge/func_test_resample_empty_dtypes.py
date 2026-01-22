from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.groupby import DataError
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import period_range
from pandas.core.indexes.timedeltas import timedelta_range
from pandas.core.resample import _asfreq_compat
@pytest.mark.parametrize('index', [PeriodIndex([], freq='M', name='a'), DatetimeIndex([], name='a'), TimedeltaIndex([], name='a')])
@pytest.mark.parametrize('dtype', [float, int, object, 'datetime64[ns]'])
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_resample_empty_dtypes(index, dtype, resample_method):
    warn = None
    if isinstance(index, PeriodIndex):
        index = PeriodIndex([], freq='B', name=index.name)
        warn = FutureWarning
    msg = 'Resampling with a PeriodIndex is deprecated'
    empty_series_dti = Series([], index, dtype)
    with tm.assert_produces_warning(warn, match=msg):
        rs = empty_series_dti.resample('d', group_keys=False)
    try:
        getattr(rs, resample_method)()
    except DataError:
        pass