import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.filterwarnings("ignore:Series.fillna with 'method' is deprecated:FutureWarning")
def test_fillna_limit_pad(self, data_missing):
    arr = data_missing.take([1, 0, 0, 0, 1])
    result = pd.Series(arr).ffill(limit=2)
    expected = pd.Series(data_missing.take([1, 1, 1, 0, 1]))
    tm.assert_series_equal(result, expected)