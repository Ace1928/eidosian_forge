from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('normalize, expected_label, expected_values', [(False, 'count', [1, 1, 1]), (True, 'proportion', [0.5, 0.5, 1.0])])
def test_mixed_groupings(normalize, expected_label, expected_values):
    df = DataFrame({'A': [1, 2, 1], 'B': [1, 2, 3]})
    gp = df.groupby([[4, 5, 4], 'A', lambda i: 7 if i == 1 else 8], as_index=False)
    result = gp.value_counts(sort=True, normalize=normalize)
    expected = DataFrame({'level_0': np.array([4, 4, 5], dtype=np.int_), 'A': [1, 1, 2], 'level_2': [8, 8, 7], 'B': [1, 3, 2], expected_label: expected_values})
    tm.assert_frame_equal(result, expected)