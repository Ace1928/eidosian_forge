from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_bool_comparison():
    df_mask = DataFrame({'AAA': [True] * 4, 'BBB': [False] * 4, 'CCC': [True, False, True, False]})
    result = df_mask.where(df_mask == False)
    expected = DataFrame({'AAA': np.array([np.nan] * 4, dtype=object), 'BBB': [False] * 4, 'CCC': [np.nan, False, np.nan, False]})
    tm.assert_frame_equal(result, expected)