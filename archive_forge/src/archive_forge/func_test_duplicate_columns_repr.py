from __future__ import annotations
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
import dask.array as da
import dask.dataframe as dd
from dask.dataframe.utils import get_string_dtype, pyarrow_strings_enabled
from dask.utils import maybe_pluralize
def test_duplicate_columns_repr():
    arr = da.from_array(np.arange(10).reshape(5, 2), chunks=(5, 2))
    frame = dd.from_dask_array(arr, columns=['a', 'a'])
    repr(frame)