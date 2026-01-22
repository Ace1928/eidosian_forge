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
def test_drop_duplicates_after_sort():
    data = [{'value': 1, 'time': 2}, {'value': 1, 'time': 1}, {'value': 2, 'time': 1}, {'value': 2, 'time': 2}]
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_df.sort_values(['value', 'time']).drop_duplicates(['value'])
    pandas_result = pandas_df.sort_values(['value', 'time']).drop_duplicates(['value'])
    sort_if_range_partitioning(modin_result, pandas_result)