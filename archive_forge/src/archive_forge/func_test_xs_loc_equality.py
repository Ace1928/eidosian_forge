import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_loc_equality(self, multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    result = df.xs(('bar', 'two'))
    expected = df.loc['bar', 'two']
    tm.assert_series_equal(result, expected)