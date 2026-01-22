import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('sort', [False, True])
@pytest.mark.parametrize('join', ['inner', 'outer'])
@pytest.mark.parametrize('axis', [0, 1])
def test_sort_order(sort, join, axis):
    pandas_df = pandas.DataFrame({'c': [3], 'd': [4]}, columns=['d', 'c'])
    pandas_df2 = pandas.DataFrame({'a': [1], 'b': [2]}, columns=['b', 'a'])
    modin_df, modin_df2 = (from_pandas(pandas_df), from_pandas(pandas_df2))
    pandas_concat = pandas.concat([pandas_df, pandas_df2], join=join, sort=sort)
    modin_concat = pd.concat([modin_df, modin_df2], join=join, sort=sort)
    df_equals(pandas_concat, modin_concat, check_dtypes=join != 'inner')
    assert list(pandas_concat.columns) == list(modin_concat.columns)