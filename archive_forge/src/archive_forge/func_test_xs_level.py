import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_level(self, multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    result = df.xs('two', level='second')
    expected = df[df.index.get_level_values(1) == 'two']
    expected.index = Index(['foo', 'bar', 'baz', 'qux'], name='first')
    tm.assert_frame_equal(result, expected)