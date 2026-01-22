import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('df1_val,df2_val,diff_self,diff_other', [(4, 3, 4, 3), (4, 4, pd.NA, pd.NA), (4, pd.NA, 4, pd.NA), (pd.NA, pd.NA, pd.NA, pd.NA)])
def test_compare_nullable_int64_dtype(df1_val, df2_val, diff_self, diff_other):
    df1 = pd.DataFrame({'a': pd.Series([df1_val, pd.NA], dtype='Int64'), 'b': [1.0, 2]})
    df2 = df1.copy()
    df2.loc[0, 'a'] = df2_val
    expected = pd.DataFrame({('a', 'self'): pd.Series([diff_self, pd.NA], dtype='Int64'), ('a', 'other'): pd.Series([diff_other, pd.NA], dtype='Int64'), ('b', 'self'): np.nan, ('b', 'other'): np.nan})
    result = df1.compare(df2, keep_shape=True)
    tm.assert_frame_equal(result, expected)