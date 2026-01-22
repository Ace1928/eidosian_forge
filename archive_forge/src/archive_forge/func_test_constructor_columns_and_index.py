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
def test_constructor_columns_and_index():
    modin_df = pd.DataFrame([[1, 1, 10], [2, 4, 20], [3, 7, 30]], index=[1, 2, 3], columns=['id', 'max_speed', 'health'])
    pandas_df = pandas.DataFrame([[1, 1, 10], [2, 4, 20], [3, 7, 30]], index=[1, 2, 3], columns=['id', 'max_speed', 'health'])
    df_equals(modin_df, pandas_df)
    df_equals(pd.DataFrame(modin_df), pandas.DataFrame(pandas_df))
    df_equals(pd.DataFrame(modin_df, columns=['max_speed', 'health']), pandas.DataFrame(pandas_df, columns=['max_speed', 'health']))
    df_equals(pd.DataFrame(modin_df, index=[1, 2]), pandas.DataFrame(pandas_df, index=[1, 2]))
    df_equals(pd.DataFrame(modin_df, index=[1, 2], columns=['health']), pandas.DataFrame(pandas_df, index=[1, 2], columns=['health']))
    df_equals(pd.DataFrame(modin_df.iloc[:, 0], index=[1, 2, 3]), pandas.DataFrame(pandas_df.iloc[:, 0], index=[1, 2, 3]))
    df_equals(pd.DataFrame(modin_df.iloc[:, 0], columns=['NO_EXIST']), pandas.DataFrame(pandas_df.iloc[:, 0], columns=['NO_EXIST']))
    with pytest.raises(NotImplementedError):
        pd.DataFrame(modin_df, index=[1, 2, 99999])
    with pytest.raises(NotImplementedError):
        pd.DataFrame(modin_df, columns=['NO_EXIST'])