from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
@pytest.fixture
def dtype_backend_expected():

    def func(storage, dtype_backend, conn_name) -> DataFrame:
        string_array: StringArray | ArrowStringArray
        string_array_na: StringArray | ArrowStringArray
        if storage == 'python':
            string_array = StringArray(np.array(['a', 'b', 'c'], dtype=np.object_))
            string_array_na = StringArray(np.array(['a', 'b', pd.NA], dtype=np.object_))
        elif dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            from pandas.arrays import ArrowExtensionArray
            string_array = ArrowExtensionArray(pa.array(['a', 'b', 'c']))
            string_array_na = ArrowExtensionArray(pa.array(['a', 'b', None]))
        else:
            pa = pytest.importorskip('pyarrow')
            string_array = ArrowStringArray(pa.array(['a', 'b', 'c']))
            string_array_na = ArrowStringArray(pa.array(['a', 'b', None]))
        df = DataFrame({'a': Series([1, np.nan, 3], dtype='Int64'), 'b': Series([1, 2, 3], dtype='Int64'), 'c': Series([1.5, np.nan, 2.5], dtype='Float64'), 'd': Series([1.5, 2.0, 2.5], dtype='Float64'), 'e': Series([True, False, pd.NA], dtype='boolean'), 'f': Series([True, False, True], dtype='boolean'), 'g': string_array, 'h': string_array_na})
        if dtype_backend == 'pyarrow':
            pa = pytest.importorskip('pyarrow')
            from pandas.arrays import ArrowExtensionArray
            df = DataFrame({col: ArrowExtensionArray(pa.array(df[col], from_pandas=True)) for col in df.columns})
        if 'mysql' in conn_name or 'sqlite' in conn_name:
            if dtype_backend == 'numpy_nullable':
                df = df.astype({'e': 'Int64', 'f': 'Int64'})
            else:
                df = df.astype({'e': 'int64[pyarrow]', 'f': 'int64[pyarrow]'})
        return df
    return func