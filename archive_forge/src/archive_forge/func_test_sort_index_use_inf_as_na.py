import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_sort_index_use_inf_as_na(self):
    expected = DataFrame({'col1': [1, 2, 3], 'col2': [3, 4, 5]}, index=pd.date_range('2020', periods=3))
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            result = expected.sort_index()
    tm.assert_frame_equal(result, expected)