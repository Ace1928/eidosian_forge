from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_empty_frame_with_datetime64_multiindex_from_groupby(using_infer_string):
    dti = pd.DatetimeIndex(['2020-01-01'], dtype='M8[ns]')
    df = DataFrame({'c1': [10.0], 'c2': ['a'], 'c3': dti})
    df = df.head(0).groupby(['c2', 'c3'])[['c1']].sum()
    result = df.reset_index()
    expected = DataFrame(columns=['c2', 'c3', 'c1'], index=RangeIndex(start=0, stop=0, step=1))
    expected['c3'] = expected['c3'].astype('datetime64[ns]')
    expected['c1'] = expected['c1'].astype('float64')
    if using_infer_string:
        expected['c2'] = expected['c2'].astype('string[pyarrow_numpy]')
    tm.assert_frame_equal(result, expected)