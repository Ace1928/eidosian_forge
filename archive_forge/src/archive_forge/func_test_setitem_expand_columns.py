import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_expand_columns(self, data):
    df = pd.DataFrame({'A': data})
    result = df.copy()
    result['B'] = 1
    expected = pd.DataFrame({'A': data, 'B': [1] * len(data)})
    tm.assert_frame_equal(result, expected)
    result = df.copy()
    result.loc[:, 'B'] = 1
    tm.assert_frame_equal(result, expected)
    result['B'] = data
    expected = pd.DataFrame({'A': data, 'B': data})
    tm.assert_frame_equal(result, expected)