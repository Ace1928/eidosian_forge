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
@pytest.mark.parametrize('testfunc', test_func_values, ids=test_func_keys)
def test_applymap_numeric(request, data, testfunc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_df.applymap(testfunc)
        except Exception as err:
            with pytest.raises(type(err)):
                modin_df.applymap(testfunc)
        else:
            modin_result = modin_df.applymap(testfunc)
            df_equals(modin_result, pandas_result)