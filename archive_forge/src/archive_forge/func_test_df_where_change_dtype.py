from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_df_where_change_dtype(self):
    df = DataFrame(np.arange(2 * 3).reshape(2, 3), columns=list('ABC'))
    mask = np.array([[True, False, False], [False, False, True]])
    result = df.where(mask)
    expected = DataFrame([[0, np.nan, np.nan], [np.nan, np.nan, 5]], columns=list('ABC'))
    tm.assert_frame_equal(result, expected)