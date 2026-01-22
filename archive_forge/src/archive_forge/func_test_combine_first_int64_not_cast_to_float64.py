from datetime import datetime
import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_combine_first_int64_not_cast_to_float64():
    df_1 = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_2 = DataFrame({'A': [1, 20, 30], 'B': [40, 50, 60], 'C': [12, 34, 65]})
    result = df_1.combine_first(df_2)
    expected = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [12, 34, 65]})
    tm.assert_frame_equal(result, expected)