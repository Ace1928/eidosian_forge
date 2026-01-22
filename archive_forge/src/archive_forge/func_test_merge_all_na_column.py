from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_all_na_column(self, series_of_dtype, series_of_dtype_all_na):
    df_left = DataFrame({'key': series_of_dtype, 'value': series_of_dtype_all_na}, columns=['key', 'value'])
    df_right = DataFrame({'key': series_of_dtype, 'value': series_of_dtype_all_na}, columns=['key', 'value'])
    expected = DataFrame({'key': series_of_dtype, 'value_x': series_of_dtype_all_na, 'value_y': series_of_dtype_all_na}, columns=['key', 'value_x', 'value_y'])
    actual = df_left.merge(df_right, on='key')
    tm.assert_frame_equal(actual, expected)