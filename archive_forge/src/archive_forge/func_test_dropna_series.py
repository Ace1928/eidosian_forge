import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_dropna_series(self, data_missing):
    ser = pd.Series(data_missing)
    result = ser.dropna()
    expected = ser.iloc[[1]]
    tm.assert_series_equal(result, expected)