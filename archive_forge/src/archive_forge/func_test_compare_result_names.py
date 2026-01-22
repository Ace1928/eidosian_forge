import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_compare_result_names():
    df1 = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1.0, 2.0, np.nan], 'col3': [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({'col1': ['c', 'b', 'c'], 'col2': [1.0, 2.0, np.nan], 'col3': [1.0, 2.0, np.nan]})
    result = df1.compare(df2, result_names=('left', 'right'))
    expected = pd.DataFrame({('col1', 'left'): {0: 'a', 2: np.nan}, ('col1', 'right'): {0: 'c', 2: np.nan}, ('col3', 'left'): {0: np.nan, 2: 3.0}, ('col3', 'right'): {0: np.nan, 2: np.nan}})
    tm.assert_frame_equal(result, expected)