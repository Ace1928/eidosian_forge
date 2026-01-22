from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_series_slicing(self):
    df = DataFrame({'a': range(3), 'b': range(4, 7)})
    result = df.where(df['a'] == 1)
    expected = df[df['a'] == 1].reindex(df.index)
    tm.assert_frame_equal(result, expected)