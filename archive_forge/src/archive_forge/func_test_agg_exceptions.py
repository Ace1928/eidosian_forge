import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.parametrize('operation', ['quantile', 'mean', 'sum', 'median', 'cumprod'])
def test_agg_exceptions(operation):
    N = 256
    fill_data = [('nan_column', [np.datetime64('2010'), None, np.datetime64('2007'), np.datetime64('2010'), np.datetime64('2006'), np.datetime64('2012'), None, np.datetime64('2011')] * (N // 8)), ('date_column', [np.datetime64('2010'), np.datetime64('2011'), np.datetime64('2011-06-15T00:00'), np.datetime64('2009-01-01')] * (N // 4))]
    data1 = {'column_to_by': ['foo', 'bar', 'baz', 'bar'] * (N // 4), 'nan_column': [np.nan] * N}
    data2 = {f'{key}{i}': value for key, value in fill_data for i in range(N // len(fill_data))}
    data = {**data1, **data2}

    def comparator(df1, df2):
        from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
        if GroupBy.is_transformation_kernel(operation):
            df1, df2 = sort_if_experimental_groupby(df1, df2)
        df_equals(df1, df2)
    expected_exception = None
    if operation == 'sum':
        expected_exception = TypeError('datetime64 type does not support sum operations')
    elif operation == 'cumprod':
        expected_exception = TypeError('datetime64 type does not support cumprod operations')
    eval_aggregation(*create_test_dfs(data), operation=operation, comparator=comparator, expected_exception=expected_exception)