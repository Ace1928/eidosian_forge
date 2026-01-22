import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_groupby_column_index_in_references():
    df = DataFrame({'A': ['a', 'b', 'c', 'd'], 'B': [1, 2, 3, 4], 'C': ['a', 'a', 'b', 'b']})
    df = df.set_index('A')
    key = df['C']
    result = df.groupby(key, observed=True).sum()
    expected = df.groupby('C', observed=True).sum()
    tm.assert_frame_equal(result, expected)