from __future__ import annotations
import contextlib
import operator
import warnings
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import _concat
from dask.dataframe.utils import (
def test_return_type_known_categories():
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    df['A'] = df['A'].astype('category')
    dask_df = dd.from_pandas(df, 2)
    ret_type = dask_df.A.cat.as_known()
    assert isinstance(ret_type, dd.Series)