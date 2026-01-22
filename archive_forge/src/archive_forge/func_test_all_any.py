import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('method', ['all', 'any'])
@pytest.mark.parametrize('is_transposed', [False, True])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('axis', axis_values, ids=axis_keys)
@pytest.mark.parametrize('data', [test_data['float_nan_data']])
def test_all_any(data, axis, skipna, is_transposed, method):
    eval_general(*create_test_dfs(data), lambda df: getattr(df.T if is_transposed else df, method)(axis=axis, skipna=skipna, bool_only=None))