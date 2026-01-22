import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
@pytest.mark.parametrize('td', [pd.Timedelta(0), pd.Timedelta('10s')])
def test_to_csv_with_dst_transitions(self, td):
    with tm.ensure_clean('csv_date_format_with_dst') as path:
        times = date_range('2013-10-26 23:00', '2013-10-27 01:00', tz='Europe/London', freq='h', ambiguous='infer')
        i = times + td
        i = i._with_freq(None)
        time_range = np.array(range(len(i)), dtype='int64')
        df = DataFrame({'A': time_range}, index=i)
        df.to_csv(path, index=True)
        result = read_csv(path, index_col=0)
        result.index = to_datetime(result.index, utc=True).tz_convert('Europe/London')
        tm.assert_frame_equal(result, df)