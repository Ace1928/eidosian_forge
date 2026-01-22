import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_single_column_slice():
    df = DataFrame('string', index=list('abcd'), columns=MultiIndex.from_product([['Main'], ('another', 'one')]))
    df['labels'] = 'a'
    df.loc[:, 'labels'] = df.index
    tm.assert_numpy_array_equal(np.asarray(df['labels']), np.asarray(df.index))
    df = DataFrame(np.nan, index=range(4), columns=MultiIndex.from_tuples([('A', '1'), ('A', '2'), ('B', '1')]))
    expected = df.copy()
    df.loc[:, 'B'] = np.arange(4)
    expected.iloc[:, 2] = np.arange(4)
    tm.assert_frame_equal(df, expected)