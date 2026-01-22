from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from dask.array import Array, from_array
from dask.dataframe import Series, _dask_expr_enabled, from_pandas, to_numeric
from dask.dataframe.utils import pyarrow_strings_enabled
from dask.delayed import Delayed
def test_to_numeric_on_dask_array_with_meta():
    arg = from_array(['1.0', '2', '-3', '5.1'])
    expected = np.array([1.0, 2.0, -3.0, 5.1])
    output = to_numeric(arg, meta=np.array((), dtype='float64'))
    expected_dtype = 'float64'
    assert output.dtype == expected_dtype
    assert isinstance(output, Array)
    assert list(output.compute()) == list(expected)