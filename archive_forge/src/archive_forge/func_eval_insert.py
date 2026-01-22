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
def eval_insert(modin_df, pandas_df, **kwargs):
    if 'col' in kwargs and 'column' not in kwargs:
        kwargs['column'] = kwargs.pop('col')
    _kwargs = {'loc': 0, 'column': 'New column'}
    _kwargs.update(kwargs)
    eval_general(modin_df, pandas_df, operation=lambda df, **kwargs: df.insert(**kwargs), __inplace__=True, **_kwargs)