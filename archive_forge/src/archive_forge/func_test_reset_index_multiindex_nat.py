from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_multiindex_nat():
    idx = range(3)
    tstamp = date_range('2015-07-01', freq='D', periods=3)
    df = DataFrame({'id': idx, 'tstamp': tstamp, 'a': list('abc')})
    df.loc[2, 'tstamp'] = pd.NaT
    result = df.set_index(['id', 'tstamp']).reset_index('id')
    exp_dti = pd.DatetimeIndex(['2015-07-01', '2015-07-02', 'NaT'], dtype='M8[ns]', name='tstamp')
    expected = DataFrame({'id': range(3), 'a': list('abc')}, index=exp_dti)
    tm.assert_frame_equal(result, expected)