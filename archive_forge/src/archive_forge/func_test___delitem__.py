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
def test___delitem__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    if 'empty_data' not in request.node.name:
        key = pandas_df.columns[0]
        modin_df = modin_df.copy()
        pandas_df = pandas_df.copy()
        modin_df.__delitem__(key)
        pandas_df.__delitem__(key)
        df_equals(modin_df, pandas_df)
        last_label = pandas_df.iloc[:, -1].name
        modin_df.__delitem__(last_label)
        pandas_df.__delitem__(last_label)
        df_equals(modin_df, pandas_df)