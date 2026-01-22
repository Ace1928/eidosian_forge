import numpy as np
import pandas as pd
import pandas._testing as tm
def test_no_mutate_but_looks_like():
    df = pd.DataFrame({'key': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'value': range(9)})
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result1 = df.groupby('key', group_keys=True).apply(lambda x: x[:].key)
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result2 = df.groupby('key', group_keys=True).apply(lambda x: x.key)
    tm.assert_series_equal(result1, result2)