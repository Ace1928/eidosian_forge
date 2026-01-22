import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame_empty_consistencies(self, using_infer_string):
    df = DataFrame(columns=['x', 'y'])
    df['x'] = [1, 2]
    expected = DataFrame({'x': [1, 2], 'y': [np.nan, np.nan]})
    tm.assert_frame_equal(df, expected, check_dtype=False)
    df = DataFrame(columns=['x', 'y'])
    df['x'] = ['1', '2']
    expected = DataFrame({'x': Series(['1', '2'], dtype=object if not using_infer_string else 'string[pyarrow_numpy]'), 'y': Series([np.nan, np.nan], dtype=object)})
    tm.assert_frame_equal(df, expected)
    df = DataFrame(columns=['x', 'y'])
    df.loc[0, 'x'] = 1
    expected = DataFrame({'x': [1], 'y': [np.nan]})
    tm.assert_frame_equal(df, expected, check_dtype=False)