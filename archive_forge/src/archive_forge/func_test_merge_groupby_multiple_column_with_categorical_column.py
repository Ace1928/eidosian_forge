import datetime
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.merge import MergeError
def test_merge_groupby_multiple_column_with_categorical_column(self):
    df = pd.DataFrame({'x': [0], 'y': [0], 'z': pd.Categorical([0])})
    result = merge_asof(df, df, on='x', by=['y', 'z'])
    expected = pd.DataFrame({'x': [0], 'y': [0], 'z': pd.Categorical([0])})
    tm.assert_frame_equal(result, expected)