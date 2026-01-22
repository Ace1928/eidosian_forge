import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.groupby.groupby import get_groupby
@pytest.mark.parametrize('f, expected_val', [['corr', 1], ['cov', 0.5]])
def test_rolling_corr_cov_other_same_size_as_groups(self, f, expected_val):
    df = DataFrame({'value': range(10), 'idx1': [1] * 5 + [2] * 5, 'idx2': [1, 2, 3, 4, 5] * 2}).set_index(['idx1', 'idx2'])
    other = DataFrame({'value': range(5), 'idx2': [1, 2, 3, 4, 5]}).set_index('idx2')
    result = getattr(df.groupby(level=0).rolling(2), f)(other)
    expected_data = ([np.nan] + [expected_val] * 4) * 2
    expected = DataFrame(expected_data, columns=['value'], index=MultiIndex.from_arrays([[1] * 5 + [2] * 5, [1] * 5 + [2] * 5, list(range(1, 6)) * 2], names=['idx1', 'idx1', 'idx2']))
    tm.assert_frame_equal(result, expected)