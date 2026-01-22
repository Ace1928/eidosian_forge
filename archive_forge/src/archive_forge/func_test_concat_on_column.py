import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.utils import from_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('no_dup_cols', [True, False])
@pytest.mark.parametrize('different_len', [True, False])
def test_concat_on_column(no_dup_cols, different_len):
    df, df2 = generate_dfs()
    if no_dup_cols:
        df = df.drop(set(df.columns) & set(df2.columns), axis='columns')
    if different_len:
        df = pandas.concat([df, df], ignore_index=True)
    modin_df, modin_df2 = (from_pandas(df), from_pandas(df2))
    df_equals(pd.concat([modin_df, modin_df2], axis=1), pandas.concat([df, df2], axis=1))
    df_equals(pd.concat([modin_df, modin_df2], axis='columns'), pandas.concat([df, df2], axis='columns'))
    modin_result = pd.concat([pd.Series(np.ones(10)), pd.Series(np.ones(10))], axis=1, ignore_index=True)
    pandas_result = pandas.concat([pandas.Series(np.ones(10)), pandas.Series(np.ones(10))], axis=1, ignore_index=True)
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)