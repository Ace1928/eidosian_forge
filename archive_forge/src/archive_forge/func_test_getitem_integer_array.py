import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('idx', [[0, 1, 2], pd.array([0, 1, 2], dtype='Int64'), np.array([0, 1, 2])], ids=['list', 'integer-array', 'numpy-array'])
def test_getitem_integer_array(self, data, idx):
    result = data[idx]
    assert len(result) == 3
    assert isinstance(result, type(data))
    expected = data.take([0, 1, 2])
    tm.assert_extension_array_equal(result, expected)
    expected = pd.Series(expected)
    result = pd.Series(data)[idx]
    tm.assert_series_equal(result, expected)