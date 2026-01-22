import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_multiindex_ints(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
    expected = df.iloc[[0, 1]].droplevel(0)
    result = df.loc[4]
    tm.assert_frame_equal(result, expected)