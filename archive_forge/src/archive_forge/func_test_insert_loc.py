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
@pytest.mark.parametrize('loc', [-3, 0, 3])
def test_insert_loc(data, loc):
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    expected_exception = None
    if loc == -3:
        expected_exception = ValueError('unbounded slice')
    eval_insert(modin_df, pandas_df, loc=loc, value=lambda df: df.iloc[:, 0], expected_exception=expected_exception)