from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from dask.array import Array, from_array
from dask.dataframe import Series, _dask_expr_enabled, from_pandas, to_numeric
from dask.dataframe.utils import pyarrow_strings_enabled
from dask.delayed import Delayed
@pytest.mark.parametrize('arg', ['5', 5, '5 '])
def test_to_numeric_on_scalars(arg):
    output = to_numeric(arg)
    assert isinstance(output, Delayed)
    assert output.compute() == 5