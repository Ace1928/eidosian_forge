from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('category', [False, True])
def test_apply_multi_level_name(category):
    b = [1, 2] * 5
    if category:
        b = pd.Categorical(b, categories=[1, 2, 3])
        expected_index = pd.CategoricalIndex([1, 2, 3], categories=[1, 2, 3], name='B')
        expected_values = [20, 25, 0]
    else:
        expected_index = Index([1, 2], name='B')
        expected_values = [20, 25]
    expected = DataFrame({'C': expected_values, 'D': expected_values}, index=expected_index)
    df = DataFrame({'A': np.arange(10), 'B': b, 'C': list(range(10)), 'D': list(range(10))}).set_index(['A', 'B'])
    result = df.groupby('B', observed=False).apply(lambda x: x.sum())
    tm.assert_frame_equal(result, expected)
    assert df.index.names == ['A', 'B']