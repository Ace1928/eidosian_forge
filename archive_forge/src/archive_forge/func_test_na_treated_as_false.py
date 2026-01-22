import pytest
import pandas as pd
import pandas._testing as tm
def test_na_treated_as_false(frame_or_series, indexer_sli):
    obj = frame_or_series([1, 2, 3])
    mask = pd.array([True, False, None], dtype='boolean')
    result = indexer_sli(obj)[mask]
    expected = indexer_sli(obj)[mask.fillna(False)]
    tm.assert_equal(result, expected)