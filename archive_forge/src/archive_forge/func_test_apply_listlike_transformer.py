import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.mark.parametrize('ops, names', [([np.sqrt], ['sqrt']), ([np.abs, np.sqrt], ['absolute', 'sqrt']), (np.array([np.sqrt]), ['sqrt']), (np.array([np.abs, np.sqrt]), ['absolute', 'sqrt'])])
def test_apply_listlike_transformer(string_series, ops, names, by_row):
    with np.errstate(all='ignore'):
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        result = string_series.apply(ops, by_row=by_row)
        tm.assert_frame_equal(result, expected)