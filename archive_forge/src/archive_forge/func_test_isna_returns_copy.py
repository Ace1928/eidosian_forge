import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('na_func', ['isna', 'notna'])
def test_isna_returns_copy(self, data_missing, na_func):
    result = pd.Series(data_missing)
    expected = result.copy()
    mask = getattr(result, na_func)()
    if isinstance(mask.dtype, pd.SparseDtype):
        mask = np.array(mask)
    mask[:] = True
    tm.assert_series_equal(result, expected)