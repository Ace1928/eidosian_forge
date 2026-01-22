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
def test_drop_transpose(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_df.T.drop(columns=[0, 1, 2])
    pandas_result = pandas_df.T.drop(columns=[0, 1, 2])
    df_equals(modin_result, pandas_result)
    modin_result = modin_df.T.drop(index=['col3', 'col1'])
    pandas_result = pandas_df.T.drop(index=['col3', 'col1'])
    df_equals(modin_result, pandas_result)
    modin_result = modin_df.T.drop(columns=[0, 1, 2], index=['col3', 'col1'])
    pandas_result = pandas_df.T.drop(columns=[0, 1, 2], index=['col3', 'col1'])
    df_equals(modin_result, pandas_result)