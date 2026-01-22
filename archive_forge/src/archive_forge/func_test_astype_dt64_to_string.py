import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_dt64_to_string(self, frame_or_series, tz_naive_fixture, using_infer_string):
    tz = tz_naive_fixture
    dti = date_range('2016-01-01', periods=3, tz=tz)
    dta = dti._data
    dta[0] = NaT
    obj = frame_or_series(dta)
    result = obj.astype('string')
    expected = frame_or_series(dta.astype('string'))
    tm.assert_equal(result, expected)
    item = result.iloc[0]
    if frame_or_series is DataFrame:
        item = item.iloc[0]
    if using_infer_string:
        assert item is np.nan
    else:
        assert item is pd.NA
    alt = obj.astype(str)
    assert np.all(alt.iloc[1:] == result.iloc[1:])