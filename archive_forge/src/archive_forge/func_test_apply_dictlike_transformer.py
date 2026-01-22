import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('ops', [{'A': np.sqrt}, {'A': np.sqrt, 'B': np.exp}, Series({'A': np.sqrt}), Series({'A': np.sqrt, 'B': np.exp})])
def test_apply_dictlike_transformer(string_series, ops, by_row):
    with np.errstate(all='ignore'):
        expected = concat({name: op(string_series) for name, op in ops.items()})
        expected.name = string_series.name
        result = string_series.apply(ops, by_row=by_row)
        tm.assert_series_equal(result, expected)