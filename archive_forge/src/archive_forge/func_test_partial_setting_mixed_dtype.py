import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_setting_mixed_dtype(self):
    df = DataFrame([[True, 1], [False, 2]], columns=['female', 'fitness'])
    s = df.loc[1].copy()
    s.name = 2
    expected = pd.concat([df, DataFrame(s).T.infer_objects()])
    df.loc[2] = df.loc[1]
    tm.assert_frame_equal(df, expected)