import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('astype', ['category', 'int32', 'float'])
def test_insert_dtypes(data, astype, request):
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    if astype == 'category' and pandas_df.iloc[:, 0].isnull().any():
        return
    expected_exception = None
    if 'int32-float_nan_data' in request.node.callspec.id:
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7026')
    eval_insert(modin_df, pandas_df, col='TypeSaver', value=lambda df: df.iloc[:, 0].astype(astype), expected_exception=expected_exception)