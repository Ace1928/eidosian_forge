from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.skipif(PANDAS_GE_210, reason='breaks with pandas=2.1.0+')
@pytest.mark.parametrize('repartition', [None, 4])
def test_merge_column_with_nulls(repartition):
    df1 = pd.DataFrame({'a': ['0', '0', None, None, None, None, '5', '7', '15', '33']})
    df2 = pd.DataFrame({'c': ['1', '2', '3', '4'], 'b': ['0', '5', '7', '15']})
    df1_d = dd.from_pandas(df1, npartitions=4)
    df2_d = dd.from_pandas(df2, npartitions=3).set_index('b')
    if repartition:
        df2_d = df2_d.repartition(npartitions=repartition)
    pandas_result = df1.merge(df2.set_index('b'), how='left', left_on='a', right_index=True)
    dask_result = df1_d.merge(df2_d, how='left', left_on='a', right_index=True)
    assert_eq(dask_result, pandas_result)