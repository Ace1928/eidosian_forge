from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@td.skip_array_manager_not_yet_implemented
def test_reindex_date_fill_value(self):
    arr = date_range('2016-01-01', periods=6).values.reshape(3, 2)
    df = DataFrame(arr, columns=['A', 'B'], index=range(3))
    ts = df.iloc[0, 0]
    fv = ts.date()
    res = df.reindex(index=range(4), columns=['A', 'B', 'C'], fill_value=fv)
    expected = DataFrame({'A': df['A'].tolist() + [fv], 'B': df['B'].tolist() + [fv], 'C': [fv] * 4}, dtype=object)
    tm.assert_frame_equal(res, expected)
    res = df.reindex(index=range(4), fill_value=fv)
    tm.assert_frame_equal(res, expected[['A', 'B']])
    res = df.reindex(index=range(4), columns=['A', 'B', 'C'], fill_value='2016-01-01')
    expected = DataFrame({'A': df['A'].tolist() + [ts], 'B': df['B'].tolist() + [ts], 'C': [ts] * 4})
    tm.assert_frame_equal(res, expected)