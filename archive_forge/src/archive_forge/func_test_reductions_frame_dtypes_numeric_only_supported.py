from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.parametrize('func', ['sum', 'prod', 'product', 'min', 'max', 'count', 'std', 'var', 'quantile'])
def test_reductions_frame_dtypes_numeric_only_supported(func):
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2011, i, 1) for i in range(1, 8)], 'str': list('abcdefgh'), 'timedelta': pd.to_timedelta([1, 2, 3, 4, 5, 6, 7, np.nan]), 'bool': [True, False] * 4})
    npartitions = 3
    if func == 'quantile':
        df = df.drop(columns='bool')
        npartitions = 1
    ddf = dd.from_pandas(df, npartitions)
    numeric_only_false_raises = ['sum', 'prod', 'product', 'std', 'var', 'quantile']
    assert_eq(getattr(df, func)(numeric_only=True), getattr(ddf, func)(numeric_only=True))
    errors = (ValueError, TypeError) if pa is None else (ValueError, TypeError, ArrowNotImplementedError)
    if func in numeric_only_false_raises:
        with pytest.raises(errors, match="'DatetimeArray' with dtype datetime64.*|'DatetimeArray' does not implement reduction|could not convert|'ArrowStringArray' with dtype string|unsupported operand|no kernel|not supported"):
            getattr(ddf, func)(numeric_only=False)
        warning = FutureWarning
    else:
        assert_eq(getattr(df, func)(numeric_only=False), getattr(ddf, func)(numeric_only=False))
        warning = None
    if PANDAS_GE_200:
        if func in numeric_only_false_raises:
            with pytest.raises(errors, match="'DatetimeArray' with dtype datetime64.*|'DatetimeArray' does not implement reduction|could not convert|'ArrowStringArray' with dtype string|unsupported operand|no kernel|not supported"):
                getattr(ddf, func)()
        else:
            assert_eq(getattr(df, func)(), getattr(ddf, func)())
    elif PANDAS_GE_150:
        if warning is None:
            pd_result = getattr(df, func)()
            dd_result = getattr(ddf, func)()
        else:
            with pytest.warns(warning, match='The default value of numeric_only'):
                pd_result = getattr(df, func)()
            with pytest.warns(warning, match='The default value of numeric_only'):
                dd_result = getattr(ddf, func)()
        assert_eq(pd_result, dd_result)
    else:
        if func in ['quantile']:
            warning = None
        if warning is None:
            pd_result = getattr(df, func)()
            dd_result = getattr(ddf, func)()
        else:
            with pytest.warns(warning, match='Dropping of nuisance'):
                pd_result = getattr(df, func)()
            with pytest.warns(warning, match='Dropping of nuisance'):
                dd_result = getattr(ddf, func)()
        assert_eq(pd_result, dd_result)
    num_cols = ['int', 'float']
    if func != 'quantile':
        num_cols.append('bool')
    df_numerics = df[num_cols]
    ddf_numerics = ddf[num_cols]
    assert_eq(getattr(df_numerics, func)(), getattr(ddf_numerics, func)())
    assert_eq(getattr(df_numerics, func)(numeric_only=False), getattr(ddf_numerics, func)(numeric_only=False))