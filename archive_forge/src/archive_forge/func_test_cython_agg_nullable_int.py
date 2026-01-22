import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('op_name', ['count', 'sum', 'std', 'var', 'sem', 'mean', 'median', 'prod', 'min', 'max'])
def test_cython_agg_nullable_int(op_name):
    df = DataFrame({'A': ['A', 'B'] * 5, 'B': pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, pd.NA], dtype='Int64')})
    result = getattr(df.groupby('A')['B'], op_name)()
    df2 = df.assign(B=df['B'].astype('float64'))
    expected = getattr(df2.groupby('A')['B'], op_name)()
    if op_name in ('mean', 'median'):
        convert_integer = False
    else:
        convert_integer = True
    expected = expected.convert_dtypes(convert_integer=convert_integer)
    tm.assert_series_equal(result, expected)