from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_align_to_series_with_common_index_level_non_unique_cols(self):
    foo_index = Index([1, 2, 3], name='foo')
    bar_index = Index([1, 2], name='bar')
    series = Series([1, 2], index=bar_index, name='foo_series')
    df = DataFrame(np.arange(18).reshape(6, 3), index=pd.MultiIndex.from_product([foo_index, bar_index]))
    df.columns = ['cfoo', 'cbar', 'cfoo']
    expected = Series([1, 2] * 3, index=df.index, name='foo_series')
    result_left, result_right = df.align(series, axis=0)
    tm.assert_series_equal(result_right, expected)
    tm.assert_index_equal(result_left.columns, df.columns)