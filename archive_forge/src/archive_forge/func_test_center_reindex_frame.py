from functools import partial
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.slow
@pytest.mark.parametrize('roll_func', ['kurt', 'skew'])
def test_center_reindex_frame(frame, roll_func):
    s = [f'x{x:d}' for x in range(12)]
    frame_xp = getattr(frame.reindex(list(frame.index) + s).rolling(window=25), roll_func)().shift(-12).reindex(frame.index)
    frame_rs = getattr(frame.rolling(window=25, center=True), roll_func)()
    tm.assert_frame_equal(frame_xp, frame_rs)