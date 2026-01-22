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
@pytest.mark.skipif(not PANDAS_GE_200, reason='ArrowDtype not supported')
def test_std_raises_with_arrow_string_ea():
    pa = pytest.importorskip('pyarrow')
    ser = pd.Series(['a', 'b', 'c'], dtype=pd.ArrowDtype(pa.string()))
    ds = dd.from_pandas(ser, npartitions=2)
    with pytest.raises(ValueError, match='`std` not supported with string series'):
        ds.std()