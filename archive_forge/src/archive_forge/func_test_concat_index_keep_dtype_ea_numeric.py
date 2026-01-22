from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_index_keep_dtype_ea_numeric(self, any_numeric_ea_dtype):
    df1 = DataFrame([[0, 1, 1]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype))
    df2 = DataFrame([[0, 1]], columns=Index([1, 2], dtype=any_numeric_ea_dtype))
    result = concat([df1, df2], ignore_index=True, join='outer', sort=True)
    expected = DataFrame([[0, 1, 1.0], [0, 1, np.nan]], columns=Index([1, 2, 3], dtype=any_numeric_ea_dtype))
    tm.assert_frame_equal(result, expected)