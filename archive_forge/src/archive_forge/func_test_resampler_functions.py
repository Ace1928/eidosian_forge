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
@pytest.mark.parametrize('rule', ['5min'])
@pytest.mark.parametrize('axis', ['index', 'columns'])
@pytest.mark.parametrize('method', [*('count', 'sum', 'std', 'sem', 'size', 'prod', 'ohlc', 'quantile'), *('min', 'median', 'mean', 'max', 'last', 'first', 'nunique', 'var'), *('interpolate', 'asfreq', 'nearest', 'bfill', 'ffill')])
def test_resampler_functions(rule, axis, method):
    data, index = (test_data_resample['data'], test_data_resample['index'])
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)
    if axis == 'columns':
        columns = pandas.date_range('31/12/2000', periods=len(pandas_df.columns), freq='min')
        modin_df.columns = columns
        pandas_df.columns = columns
    expected_exception = None
    if method in ('interpolate', 'asfreq', 'nearest', 'bfill', 'ffill'):
        expected_exception = AssertionError('axis must be 0')
    eval_general(modin_df, pandas_df, lambda df: getattr(df.resample(rule, axis=axis), method)(), expected_exception=expected_exception)