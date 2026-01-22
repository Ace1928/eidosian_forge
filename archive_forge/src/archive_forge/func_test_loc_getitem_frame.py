import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_frame(self):
    df = DataFrame({'A': range(10)})
    ser = pd.cut(df.A, 5)
    df['B'] = ser
    df = df.set_index('B')
    result = df.loc[4]
    expected = df.iloc[4:6]
    tm.assert_frame_equal(result, expected)
    with pytest.raises(KeyError, match='10'):
        df.loc[10]
    result = df.loc[[4]]
    expected = df.iloc[4:6]
    tm.assert_frame_equal(result, expected)
    result = df.loc[[4, 5]]
    expected = df.take([4, 5, 4, 5])
    tm.assert_frame_equal(result, expected)
    msg = "None of \\[Index\\(\\[10\\], dtype='object', name='B'\\)\\] are in the \\[index\\]"
    with pytest.raises(KeyError, match=msg):
        df.loc[[10]]
    with pytest.raises(KeyError, match='\\[10\\] not in index'):
        df.loc[[10, 4]]