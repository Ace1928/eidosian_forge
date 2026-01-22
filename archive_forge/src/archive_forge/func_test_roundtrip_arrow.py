from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@PYARROW_MARK
@pytest.mark.parametrize('df', [pd.DataFrame({'x': [4, 5, 6, 1, 2, 3]}), pd.DataFrame({'x': ['c', 'a', 'b']}), pd.DataFrame({'x': ['cc', 'a', 'bbb']}), pytest.param(pd.DataFrame({'x': pd.Categorical(['a', 'b', 'a'])})), pytest.param(pd.DataFrame({'x': pd.Categorical([1, 2, 1])})), pd.DataFrame({'x': list(map(pd.Timestamp, [3000000, 2000000, 1000000]))}), pd.DataFrame({'x': list(map(pd.Timestamp, [3000, 2000, 1000]))}), pd.DataFrame({'x': [3000, 2000, 1000]}).astype('M8[ns]'), pytest.param(pd.DataFrame({'x': [3, 2, 1]}).astype('M8[us]'), marks=pytest.mark.xfail(PANDAS_GE_200 and pyarrow_version < parse_version('13.0.0.dev'), reason='https://github.com/apache/arrow/issues/15079')), pytest.param(pd.DataFrame({'x': [3, 2, 1]}).astype('M8[ms]'), marks=pytest.mark.xfail(PANDAS_GE_200 and pyarrow_version < parse_version('13.0.0.dev'), reason='https://github.com/apache/arrow/issues/15079')), pd.DataFrame({'x': [3, 2, 1]}).astype('uint16'), pd.DataFrame({'x': [3, 2, 1]}).astype('float32'), pd.DataFrame({'x': [3, 1, 2]}, index=[3, 2, 1]), pd.DataFrame({'x': [4, 5, 6, 1, 2, 3]}, index=pd.Index([1, 2, 3, 4, 5, 6], name='foo')), pd.DataFrame({'x': [1, 2, 3], 'y': [3, 2, 1]}), pd.DataFrame({'x': [1, 2, 3], 'y': [3, 2, 1]}, columns=['y', 'x']), pd.DataFrame({'0': [3, 2, 1]}), pd.DataFrame({'x': [3, 2, None]}), pd.DataFrame({'-': [3.0, 2.0, None]}), pd.DataFrame({'.': [3.0, 2.0, None]}), pd.DataFrame({' ': [3.0, 2.0, None]})])
def test_roundtrip_arrow(tmpdir, df):
    tmp_path = str(tmpdir)
    if not df.index.name:
        df.index.name = 'index'
    ddf = dd.from_pandas(df, npartitions=2)
    dd.to_parquet(ddf, tmp_path, write_index=True)
    ddf2 = dd.read_parquet(tmp_path, calculate_divisions=True)
    assert_eq(ddf, ddf2)