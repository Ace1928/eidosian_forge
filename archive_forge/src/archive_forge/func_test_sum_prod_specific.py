import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('fn', ['prod', 'sum'])
@pytest.mark.parametrize('numeric_only', [False, True])
@pytest.mark.parametrize('min_count', int_arg_values, ids=arg_keys('min_count', int_arg_keys))
def test_sum_prod_specific(fn, min_count, numeric_only):
    expected_exception = None
    if not numeric_only and fn == 'prod':
        expected_exception = False
    elif not numeric_only and fn == 'sum':
        expected_exception = TypeError('can only concatenate str (not "int") to str')
    if numeric_only and fn == 'sum':
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7029')
    if min_count == 5 and (not numeric_only):
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7029')
    eval_general(*create_test_dfs(test_data_diff_dtype), lambda df: getattr(df, fn)(min_count=min_count, numeric_only=numeric_only), expected_exception=expected_exception)