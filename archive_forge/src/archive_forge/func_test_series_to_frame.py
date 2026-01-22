import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_to_frame(self):
    orig = Series([1, 2, 3])
    expected = orig.to_frame()
    result = SimpleSeriesSubClass(orig).to_frame()
    assert type(result) is DataFrame
    tm.assert_frame_equal(result, expected)