import numpy as np
import pytest
from pandas.errors import SettingWithCopyWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def test_set_index_mutating_parent_does_not_mutate_index():
    df = DataFrame({'a': [1, 2, 3], 'b': 1})
    result = df.set_index('a')
    expected = result.copy()
    df.iloc[0, 0] = 100
    tm.assert_frame_equal(result, expected)