from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('df1_vals, df2_vals', [(Series([1, 2], dtype='uint64'), ['a', 'b', 'c']), (Series([1, 2], dtype='int32'), ['a', 'b', 'c']), ([0, 1, 2], ['0', '1', '2']), ([0.0, 1.0, 2.0], ['0', '1', '2']), ([0, 1, 2], ['0', '1', '2']), (pd.date_range('1/1/2011', periods=2, freq='D'), ['2011-01-01', '2011-01-02']), (pd.date_range('1/1/2011', periods=2, freq='D'), [0, 1]), (pd.date_range('1/1/2011', periods=2, freq='D'), [0.0, 1.0]), (pd.date_range('20130101', periods=3), pd.date_range('20130101', periods=3, tz='US/Eastern'))])
def test_merge_incompat_dtypes_error(self, df1_vals, df2_vals):
    df1 = DataFrame({'A': df1_vals})
    df2 = DataFrame({'A': df2_vals})
    msg = f"You are trying to merge on {df1['A'].dtype} and {df2['A'].dtype} columns for key 'A'. If you wish to proceed you should use pd.concat"
    msg = re.escape(msg)
    with pytest.raises(ValueError, match=msg):
        merge(df1, df2, on=['A'])
    msg = f"You are trying to merge on {df2['A'].dtype} and {df1['A'].dtype} columns for key 'A'. If you wish to proceed you should use pd.concat"
    msg = re.escape(msg)
    with pytest.raises(ValueError, match=msg):
        merge(df2, df1, on=['A'])
    if len(df1_vals) == len(df2_vals):
        df3 = DataFrame({'A': df2_vals, 'B': df1_vals, 'C': df1_vals})
        df4 = DataFrame({'A': df2_vals, 'B': df2_vals, 'C': df2_vals})
        msg = f"You are trying to merge on {df3['B'].dtype} and {df4['B'].dtype} columns for key 'B'. If you wish to proceed you should use pd.concat"
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df3, df4)
        msg = f"You are trying to merge on {df3['C'].dtype} and {df4['C'].dtype} columns for key 'C'. If you wish to proceed you should use pd.concat"
        msg = re.escape(msg)
        with pytest.raises(ValueError, match=msg):
            merge(df3, df4, on=['A', 'C'])