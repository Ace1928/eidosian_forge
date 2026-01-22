from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
@pytest.mark.filterwarnings("ignore:The 'convention' keyword in Series.resample:FutureWarning")
@pytest.mark.parametrize('_index_start,_index_end,_index_name', [('1/1/2000 00:00:00', '1/1/2000 00:13:00', 'index')])
@pytest.mark.parametrize('keyword,value', [('label', 'righttt'), ('closed', 'righttt'), ('convention', 'starttt')])
def test_resample_string_kwargs(series, keyword, value, unit):
    series.index = series.index.as_unit(unit)
    msg = f'Unsupported value {value} for `{keyword}`'
    with pytest.raises(ValueError, match=msg):
        series.resample('5min', **{keyword: value})