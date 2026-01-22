import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_values(self, multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    result = df.xs(('bar', 'two')).values
    expected = df.values[4]
    tm.assert_almost_equal(result, expected)