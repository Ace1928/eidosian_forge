import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_take_series(self, data):
    s = pd.Series(data)
    result = s.take([0, -1])
    expected = pd.Series(data._from_sequence([data[0], data[len(data) - 1]], dtype=s.dtype), index=[0, len(data) - 1])
    tm.assert_series_equal(result, expected)