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
@pytest.mark.parametrize('dtypes_are_dict', [True, False])
def test_astype_dict_or_series_multiple_column_partitions(dtypes_are_dict):
    modin_df, pandas_df = create_test_dfs(test_data['int_data'])
    if dtypes_are_dict:
        new_dtypes = {}
    else:
        new_dtypes = pandas.Series()
    for i, column in enumerate(pandas_df.columns):
        if i % 3 == 1:
            new_dtypes[column] = 'string'
        elif i % 3 == 2:
            new_dtypes[column] = float
    eval_general(modin_df, pandas_df, lambda df: df.astype(new_dtypes))