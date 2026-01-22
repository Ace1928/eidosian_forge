import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_corr_pairwise(frame):
    result = frame.expanding().corr()
    rolling_result = frame.rolling(window=len(frame), min_periods=1).corr()
    tm.assert_frame_equal(result, rolling_result)