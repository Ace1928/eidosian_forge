from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.tests.extension.decimal.array import DecimalDtype
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_150
from dask.dataframe._pyarrow import (
@pytest.mark.parametrize('dtype,expected', [(object, True), (str, True), (np.dtype(int), False), (np.dtype(float), False), (pd.StringDtype('python'), True), (DecimalDtype(), False), pytest.param(pa.int64(), False, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='Needs pd.ArrowDtype')), pytest.param(pa.float64(), False, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='Needs pd.ArrowDtype')), (pd.StringDtype('pyarrow'), False), pytest.param(pa.string(), False, marks=pytest.mark.skipif(not PANDAS_GE_150, reason='Needs pd.ArrowDtype'))])
def test_is_object_string_dtype(dtype, expected):
    if isinstance(dtype, pa.DataType):
        dtype = pd.ArrowDtype(dtype)
    assert is_object_string_dtype(dtype) is expected