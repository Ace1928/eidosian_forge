import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_join_5203():
    data = np.ones([2, 4])
    kwargs = {'columns': ['a', 'b', 'c', 'd']}
    modin_dfs, pandas_dfs = ([None] * 3, [None] * 3)
    for idx in range(len(modin_dfs)):
        modin_dfs[idx], pandas_dfs[idx] = create_test_dfs(data, **kwargs)
    for dfs in (modin_dfs, pandas_dfs):
        with pytest.raises(ValueError, match='Joining multiple DataFrames only supported for joining on index'):
            dfs[0].join([dfs[1], dfs[2]], how='inner', on='a')