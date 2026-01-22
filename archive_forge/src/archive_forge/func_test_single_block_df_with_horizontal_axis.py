import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_single_block_df_with_horizontal_axis(self):
    df = DataFrame({'col1': [5, 0, np.nan, 10, np.nan], 'col2': [7, np.nan, np.nan, 5, 3], 'col3': [12, np.nan, 1, 2, 0], 'col4': [np.nan, 1, 1, np.nan, 18]})
    result = df.fillna(50, limit=1, axis=1)
    expected = DataFrame([[5.0, 7.0, 12.0, 50.0], [0.0, 50.0, np.nan, 1.0], [50.0, np.nan, 1.0, 1.0], [10.0, 5.0, 2.0, 50.0], [50.0, 3.0, 0.0, 18.0]], columns=['col1', 'col2', 'col3', 'col4'])
    tm.assert_frame_equal(result, expected)