import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_setitem_zerolen_list_length_must_match_columns(self):
    df = DataFrame(columns=['A', 'B'])
    msg = 'cannot set a row with mismatched columns'
    with pytest.raises(ValueError, match=msg):
        df.loc[0] = [1, 2, 3]
    df = DataFrame(columns=['A', 'B'])
    df.loc[3] = [6, 7]
    exp = DataFrame([[6, 7]], index=[3], columns=['A', 'B'], dtype=np.int64)
    tm.assert_frame_equal(df, exp)