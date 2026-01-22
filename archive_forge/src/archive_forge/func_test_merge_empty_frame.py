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
def test_merge_empty_frame(self, series_of_dtype, series_of_dtype2):
    df = DataFrame({'key': series_of_dtype, 'value': series_of_dtype2}, columns=['key', 'value'])
    df_empty = df[:0]
    expected = DataFrame({'key': Series(dtype=df.dtypes['key']), 'value_x': Series(dtype=df.dtypes['value']), 'value_y': Series(dtype=df.dtypes['value'])}, columns=['key', 'value_x', 'value_y'])
    actual = df_empty.merge(df, on='key')
    tm.assert_frame_equal(actual, expected)