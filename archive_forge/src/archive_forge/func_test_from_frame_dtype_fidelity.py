from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_frame_dtype_fidelity():
    df = pd.DataFrame({'dates': date_range('19910905', periods=6, tz='US/Eastern'), 'a': [1, 1, 1, 2, 2, 2], 'b': pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'], ordered=True), 'c': ['x', 'x', 'y', 'z', 'x', 'y']})
    original_dtypes = df.dtypes.to_dict()
    expected_mi = MultiIndex.from_arrays([date_range('19910905', periods=6, tz='US/Eastern'), [1, 1, 1, 2, 2, 2], pd.Categorical(['a', 'a', 'b', 'b', 'c', 'c'], ordered=True), ['x', 'x', 'y', 'z', 'x', 'y']], names=['dates', 'a', 'b', 'c'])
    mi = MultiIndex.from_frame(df)
    mi_dtypes = {name: mi.levels[i].dtype for i, name in enumerate(mi.names)}
    tm.assert_index_equal(expected_mi, mi)
    assert original_dtypes == mi_dtypes