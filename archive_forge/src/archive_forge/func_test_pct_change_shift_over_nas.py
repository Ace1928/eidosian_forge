import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_pct_change_shift_over_nas(self):
    s = Series([1.0, 1.5, np.nan, 2.5, 3.0])
    df = DataFrame({'a': s, 'b': s})
    msg = "The default fill_method='pad' in DataFrame.pct_change is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        chg = df.pct_change()
    expected = Series([np.nan, 0.5, 0.0, 2.5 / 1.5 - 1, 0.2])
    edf = DataFrame({'a': expected, 'b': expected})
    tm.assert_frame_equal(chg, edf)