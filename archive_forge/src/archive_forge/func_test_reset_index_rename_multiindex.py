from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_rename_multiindex(float_frame):
    stacked_df = float_frame.stack(future_stack=True)[::2]
    stacked_df = DataFrame({'foo': stacked_df, 'bar': stacked_df})
    names = ['first', 'second']
    stacked_df.index.names = names
    result = stacked_df.reset_index()
    expected = stacked_df.reset_index(names=['new_first', 'new_second'])
    tm.assert_series_equal(result['first'], expected['new_first'], check_names=False)
    tm.assert_series_equal(result['second'], expected['new_second'], check_names=False)