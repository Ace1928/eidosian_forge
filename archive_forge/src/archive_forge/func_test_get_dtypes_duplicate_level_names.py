import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_dtypes_duplicate_level_names(using_infer_string):
    result = MultiIndex.from_product([[1, 2, 3], ['a', 'b', 'c'], pd.date_range('20200101', periods=2, tz='UTC')], names=['A', 'A', 'A']).dtypes
    exp = 'object' if not using_infer_string else 'string'
    expected = pd.Series([np.dtype('int64'), exp, DatetimeTZDtype(tz='utc')], index=['A', 'A', 'A'])
    tm.assert_series_equal(result, expected)