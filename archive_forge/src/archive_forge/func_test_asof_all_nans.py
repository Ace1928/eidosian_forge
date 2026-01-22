import numpy as np
import pytest
from pandas._libs.tslibs import IncompatibleFrequency
from pandas import (
import pandas._testing as tm
def test_asof_all_nans(self, frame_or_series):
    result = frame_or_series([np.nan]).asof([0])
    expected = frame_or_series([np.nan])
    tm.assert_equal(result, expected)