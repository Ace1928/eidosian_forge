from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_consistent_names(int_frame_const_col):
    df = int_frame_const_col
    result = df.apply(lambda x: Series([1, 2, 3], index=['test', 'other', 'cols']), axis=1)
    expected = int_frame_const_col.rename(columns={'A': 'test', 'B': 'other', 'C': 'cols'})
    tm.assert_frame_equal(result, expected)
    result = df.apply(lambda x: Series([1, 2], index=['test', 'other']), axis=1)
    expected = expected[['test', 'other']]
    tm.assert_frame_equal(result, expected)