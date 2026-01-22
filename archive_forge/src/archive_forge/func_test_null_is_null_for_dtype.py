import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('test_series', [True, False])
@pytest.mark.parametrize('dtype', [object, None])
def test_null_is_null_for_dtype(sort, dtype, nulls_fixture, nulls_fixture2, test_series):
    df = pd.DataFrame({'a': [1, 2]})
    groups = pd.Series([nulls_fixture, nulls_fixture2], dtype=dtype)
    obj = df['a'] if test_series else df
    gb = obj.groupby(groups, dropna=False, sort=sort)
    result = gb.sum()
    index = pd.Index([na_value_for_dtype(groups.dtype)])
    expected = pd.DataFrame({'a': [3]}, index=index)
    if test_series:
        tm.assert_series_equal(result, expected['a'])
    else:
        tm.assert_frame_equal(result, expected)