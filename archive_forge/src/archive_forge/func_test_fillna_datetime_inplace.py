import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_datetime_inplace(self):
    df = DataFrame({'date1': to_datetime(['2018-05-30', None]), 'date2': to_datetime(['2018-09-30', None])})
    expected = df.copy()
    df.fillna(np.nan, inplace=True)
    tm.assert_frame_equal(df, expected)