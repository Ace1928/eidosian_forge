from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('l_ordered,r_ordered,expected', [[True, True, pd.CategoricalIndex], [True, False, Index], [False, True, Index], [False, False, pd.CategoricalIndex]])
def test_align_categorical(self, l_ordered, r_ordered, expected):
    df_1 = DataFrame({'A': np.arange(6, dtype='int64'), 'B': Series(list('aabbca')).astype(pd.CategoricalDtype(list('cab'), ordered=l_ordered))}).set_index('B')
    df_2 = DataFrame({'A': np.arange(5, dtype='int64'), 'B': Series(list('babca')).astype(pd.CategoricalDtype(list('cab'), ordered=r_ordered))}).set_index('B')
    aligned_1, aligned_2 = df_1.align(df_2)
    assert isinstance(aligned_1.index, expected)
    assert isinstance(aligned_2.index, expected)
    tm.assert_index_equal(aligned_1.index, aligned_2.index)