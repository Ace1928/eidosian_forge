from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('box', [lambda x: list(x), lambda x: tuple(x), lambda x: np.array(x, dtype='int64')], ids=['list', 'tuple', 'array'])
def test_consistency_for_boxed(box, int_frame_const_col):
    df = int_frame_const_col
    result = df.apply(lambda x: box([1, 2]), axis=1)
    expected = Series([box([1, 2]) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)
    result = df.apply(lambda x: box([1, 2]), axis=1, result_type='expand')
    expected = int_frame_const_col[['A', 'B']].rename(columns={'A': 0, 'B': 1})
    tm.assert_frame_equal(result, expected)