import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('agg_func', ['any', 'all'])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('vals', [['foo', 'bar', 'baz'], ['foo', '', ''], ['', '', ''], [1, 2, 3], [1, 0, 0], [0, 0, 0], [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [True, True, True], [True, False, False], [False, False, False], [np.nan, np.nan, np.nan]])
def test_groupby_bool_aggs(agg_func, skipna, vals):
    df = DataFrame({'key': ['a'] * 3 + ['b'] * 3, 'val': vals * 2})
    exp = getattr(builtins, agg_func)(vals)
    if skipna and all(isna(vals)) and (agg_func == 'any'):
        exp = False
    exp_df = DataFrame([exp] * 2, columns=['val'], index=Index(['a', 'b'], name='key'))
    result = getattr(df.groupby('key'), agg_func)(skipna=skipna)
    tm.assert_frame_equal(result, exp_df)