import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('index', [lambda df: df.columns[0], lambda df: df.columns[:2], lib.no_default], ids=['one_column_index', 'several_columns_index', 'default'])
@pytest.mark.parametrize('columns', [lambda df: df.columns[len(df.columns) // 2]], ids=['one_column'])
@pytest.mark.parametrize('values', [lambda df: df.columns[-1], lambda df: df.columns[-2:], lib.no_default], ids=['one_column_values', 'several_columns_values', 'default'])
def test_pivot(data, index, columns, values, request):
    current_execution = get_current_execution()
    if 'one_column_values-one_column-default-float_nan_data' in request.node.callspec.id or 'default-one_column-several_columns_index' in request.node.callspec.id or 'default-one_column-one_column_index' in request.node.callspec.id or (current_execution in ('BaseOnPython', 'HdkOnNative') and index is lib.no_default):
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7010')
    expected_exception = None
    if index is not lib.no_default:
        expected_exception = ValueError('Index contains duplicate entries, cannot reshape')
    eval_general(*create_test_dfs(data), lambda df, *args, **kwargs: df.pivot(*args, **kwargs), index=index, columns=columns, values=values, expected_exception=expected_exception)