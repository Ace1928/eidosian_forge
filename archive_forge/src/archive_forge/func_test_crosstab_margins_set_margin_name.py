import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_margins_set_margin_name(self):
    a = np.random.default_rng(2).integers(0, 7, size=100)
    b = np.random.default_rng(2).integers(0, 3, size=100)
    c = np.random.default_rng(2).integers(0, 5, size=100)
    df = DataFrame({'a': a, 'b': b, 'c': c})
    result = crosstab(a, [b, c], rownames=['a'], colnames=('b', 'c'), margins=True, margins_name='TOTAL')
    assert result.index.names == ('a',)
    assert result.columns.names == ['b', 'c']
    all_cols = result['TOTAL', '']
    exp_cols = df.groupby(['a']).size().astype('i8')
    exp_margin = Series([len(df)], index=Index(['TOTAL'], name='a'))
    exp_cols = pd.concat([exp_cols, exp_margin])
    exp_cols.name = ('TOTAL', '')
    tm.assert_series_equal(all_cols, exp_cols)
    all_rows = result.loc['TOTAL']
    exp_rows = df.groupby(['b', 'c']).size().astype('i8')
    exp_rows = pd.concat([exp_rows, Series([len(df)], index=[('TOTAL', '')])])
    exp_rows.name = 'TOTAL'
    exp_rows = exp_rows.reindex(all_rows.index)
    exp_rows = exp_rows.fillna(0).astype(np.int64)
    tm.assert_series_equal(all_rows, exp_rows)
    msg = 'margins_name argument must be a string'
    for margins_name in [666, None, ['a', 'b']]:
        with pytest.raises(ValueError, match=msg):
            crosstab(a, [b, c], rownames=['a'], colnames=('b', 'c'), margins=True, margins_name=margins_name)