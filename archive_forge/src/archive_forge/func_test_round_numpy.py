import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_round_numpy(self):
    df = DataFrame([[1.53, 1.36], [0.06, 7.01]])
    out = np.round(df, decimals=0)
    expected = DataFrame([[2.0, 1.0], [0.0, 7.0]])
    tm.assert_frame_equal(out, expected)
    msg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=msg):
        np.round(df, decimals=0, out=df)