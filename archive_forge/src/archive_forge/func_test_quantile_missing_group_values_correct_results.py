import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key, val, expected_key, expected_val', [([1.0, np.nan, 3.0, np.nan], range(4), [1.0, 3.0], [0.0, 2.0]), ([1.0, np.nan, 2.0, 2.0], range(4), [1.0, 2.0], [0.0, 2.5]), (['a', 'b', 'b', np.nan], range(4), ['a', 'b'], [0, 1.5]), ([0], [42], [0], [42.0]), ([], [], np.array([], dtype='float64'), np.array([], dtype='float64'))])
def test_quantile_missing_group_values_correct_results(key, val, expected_key, expected_val):
    df = DataFrame({'key': key, 'val': val})
    expected = DataFrame(expected_val, index=Index(expected_key, name='key'), columns=['val'])
    grp = df.groupby('key')
    result = grp.quantile(0.5)
    tm.assert_frame_equal(result, expected)
    result = grp.quantile()
    tm.assert_frame_equal(result, expected)