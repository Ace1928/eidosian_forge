import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func,window_size,expected_vals', [('rolling', 2, [[np.nan, np.nan, np.nan, np.nan], [15.0, 20.0, 25.0, 20.0], [25.0, 30.0, 35.0, 30.0], [np.nan, np.nan, np.nan, np.nan], [20.0, 30.0, 35.0, 30.0], [35.0, 40.0, 60.0, 40.0], [60.0, 80.0, 85.0, 80]]), ('expanding', None, [[10.0, 10.0, 20.0, 20.0], [15.0, 20.0, 25.0, 20.0], [20.0, 30.0, 30.0, 20.0], [10.0, 10.0, 30.0, 30.0], [20.0, 30.0, 35.0, 30.0], [26.666667, 40.0, 50.0, 30.0], [40.0, 80.0, 60.0, 30.0]])])
def test_multiple_agg_funcs(func, window_size, expected_vals):
    df = DataFrame([['A', 10, 20], ['A', 20, 30], ['A', 30, 40], ['B', 10, 30], ['B', 30, 40], ['B', 40, 80], ['B', 80, 90]], columns=['stock', 'low', 'high'])
    f = getattr(df.groupby('stock'), func)
    if window_size:
        window = f(window_size)
    else:
        window = f()
    index = MultiIndex.from_tuples([('A', 0), ('A', 1), ('A', 2), ('B', 3), ('B', 4), ('B', 5), ('B', 6)], names=['stock', None])
    columns = MultiIndex.from_tuples([('low', 'mean'), ('low', 'max'), ('high', 'mean'), ('high', 'min')])
    expected = DataFrame(expected_vals, index=index, columns=columns)
    result = window.agg({'low': ['mean', 'max'], 'high': ['mean', 'min']})
    tm.assert_frame_equal(result, expected)