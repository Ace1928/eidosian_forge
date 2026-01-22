import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_agg_multiple_functions_same_name_with_ohlc_present():
    df = DataFrame(np.random.default_rng(2).standard_normal((1000, 3)), index=pd.date_range('1/1/2012', freq='s', periods=1000, name='dti'), columns=Index(['A', 'B', 'C'], name='alpha'))
    result = df.resample('3min').agg({'A': ['ohlc', partial(np.quantile, q=0.9999), partial(np.quantile, q=0.1111)]})
    expected_index = pd.date_range('1/1/2012', freq='3min', periods=6, name='dti')
    expected_columns = MultiIndex.from_tuples([('A', 'ohlc', 'open'), ('A', 'ohlc', 'high'), ('A', 'ohlc', 'low'), ('A', 'ohlc', 'close'), ('A', 'quantile', 'A'), ('A', 'quantile', 'A')], names=['alpha', None, None])
    non_ohlc_expected_values = np.array([df.resample('3min').A.quantile(q=q).values for q in [0.9999, 0.1111]]).T
    expected_values = np.hstack([df.resample('3min').A.ohlc(), non_ohlc_expected_values])
    expected = DataFrame(expected_values, columns=expected_columns, index=expected_index)
    tm.assert_frame_equal(result, expected)