import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('before, after, indices', [(1, 2, [2, 1]), (None, 2, [2, 1, 0]), (1, None, [3, 2, 1])])
@pytest.mark.parametrize('dtyp', [*tm.ALL_REAL_NUMPY_DTYPES, 'datetime64[ns]'])
def test_truncate_decreasing_index(self, before, after, indices, dtyp, frame_or_series):
    idx = Index([3, 2, 1, 0], dtype=dtyp)
    if isinstance(idx, DatetimeIndex):
        before = pd.Timestamp(before) if before is not None else None
        after = pd.Timestamp(after) if after is not None else None
        indices = [pd.Timestamp(i) for i in indices]
    values = frame_or_series(range(len(idx)), index=idx)
    result = values.truncate(before=before, after=after)
    expected = values.loc[indices]
    tm.assert_equal(result, expected)