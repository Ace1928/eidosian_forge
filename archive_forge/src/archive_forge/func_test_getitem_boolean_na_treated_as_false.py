import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_getitem_boolean_na_treated_as_false(self, data):
    mask = pd.array(np.zeros(data.shape, dtype='bool'), dtype='boolean')
    mask[:2] = pd.NA
    mask[2:4] = True
    result = data[mask]
    expected = data[mask.fillna(False)]
    tm.assert_extension_array_equal(result, expected)
    s = pd.Series(data)
    result = s[mask]
    expected = s[mask.fillna(False)]
    tm.assert_series_equal(result, expected)