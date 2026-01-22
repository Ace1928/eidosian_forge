import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('left_index', ['inner', ['inner', 'outer']])
def test_join_indexes_and_columns_on(df1, df2, left_index, join_type):
    left_df = df1.set_index(left_index)
    right_df = df2.set_index(['outer', 'inner'])
    expected = left_df.reset_index().join(right_df, on=['outer', 'inner'], how=join_type, lsuffix='_x', rsuffix='_y').set_index(left_index)
    result = left_df.join(right_df, on=['outer', 'inner'], how=join_type, lsuffix='_x', rsuffix='_y')
    tm.assert_frame_equal(result, expected, check_like=True)