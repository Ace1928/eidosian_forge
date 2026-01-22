from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_index_date(using_infer_string):
    ts = ['2011-05-16 00:00', '2011-05-16 01:00', '2011-05-16 02:00', '2011-05-16 03:00', '2011-05-17 02:00', '2011-05-17 03:00', '2011-05-17 04:00', '2011-05-17 05:00', '2011-05-18 02:00', '2011-05-18 03:00', '2011-05-18 04:00', '2011-05-18 05:00']
    df = DataFrame({'value': [1.40893, 1.4076, 1.4075, 1.40649, 1.40893, 1.4076, 1.4075, 1.40649, 1.40893, 1.4076, 1.4075, 1.40649]}, index=Index(pd.to_datetime(ts), name='date_time'))
    expected = df.groupby(df.index.date).idxmax()
    result = df.groupby(df.index.date).apply(lambda x: x.idxmax())
    tm.assert_frame_equal(result, expected)