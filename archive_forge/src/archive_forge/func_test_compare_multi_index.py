import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('align_axis', [0, 1])
def test_compare_multi_index(align_axis):
    df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1.0, 2.0, np.nan], 'col3': [1.0, 2.0, 3.0]})
    df.columns = pd.MultiIndex.from_arrays([['a', 'a', 'b'], ['col1', 'col2', 'col3']])
    df.index = pd.MultiIndex.from_arrays([['x', 'x', 'y'], [0, 1, 2]])
    df2 = df.copy()
    df2.iloc[0, 0] = 'c'
    df2.iloc[2, 2] = 4.0
    result = df.compare(df2, align_axis=align_axis)
    if align_axis == 0:
        indices = pd.MultiIndex.from_arrays([['x', 'x', 'y', 'y'], [0, 0, 2, 2], ['self', 'other', 'self', 'other']])
        columns = pd.MultiIndex.from_arrays([['a', 'b'], ['col1', 'col3']])
        data = [['a', np.nan], ['c', np.nan], [np.nan, 3.0], [np.nan, 4.0]]
    else:
        indices = pd.MultiIndex.from_arrays([['x', 'y'], [0, 2]])
        columns = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], ['col1', 'col1', 'col3', 'col3'], ['self', 'other', 'self', 'other']])
        data = [['a', 'c', np.nan, np.nan], [np.nan, np.nan, 3.0, 4.0]]
    expected = pd.DataFrame(data=data, index=indices, columns=columns)
    tm.assert_frame_equal(result, expected)