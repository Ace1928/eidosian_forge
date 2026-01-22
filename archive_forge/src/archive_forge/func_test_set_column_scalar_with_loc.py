import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_column_scalar_with_loc(self, multiindex_dataframe_random_data, using_copy_on_write, warn_copy_on_write):
    frame = multiindex_dataframe_random_data
    subset = frame.index[[1, 4, 5]]
    frame.loc[subset] = 99
    assert (frame.loc[subset].values == 99).all()
    frame_original = frame.copy()
    col = frame['B']
    with tm.assert_cow_warning(warn_copy_on_write):
        col[subset] = 97
    if using_copy_on_write:
        tm.assert_frame_equal(frame, frame_original)
    else:
        assert (frame.loc[subset, 'B'] == 97).all()