from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_setitem_categorical_missing(self):
    df = DataFrame(index=range(3), columns=range(3), dtype=CategoricalDtype(['foo', 'bar']))
    df.at[1, 1] = 'foo'
    expected = DataFrame([[np.nan, np.nan, np.nan], [np.nan, 'foo', np.nan], [np.nan, np.nan, np.nan]], dtype=CategoricalDtype(['foo', 'bar']))
    tm.assert_frame_equal(df, expected)