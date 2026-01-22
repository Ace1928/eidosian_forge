import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_dropna_frame(self, data_missing):
    df = pd.DataFrame({'A': data_missing}, columns=pd.Index(['A'], dtype=object))
    result = df.dropna()
    expected = df.iloc[[1]]
    tm.assert_frame_equal(result, expected)
    result = df.dropna(axis='columns')
    expected = pd.DataFrame(index=pd.RangeIndex(2), columns=pd.Index([]))
    tm.assert_frame_equal(result, expected)
    df = pd.DataFrame({'A': data_missing, 'B': [1, np.nan]})
    result = df.dropna()
    expected = df.iloc[:0]
    tm.assert_frame_equal(result, expected)