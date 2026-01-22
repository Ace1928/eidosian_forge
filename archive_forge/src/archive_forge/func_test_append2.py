from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
@pytest.mark.skipif(PANDAS_GE_200, reason='pandas removed append')
def test_append2():
    dsk = {('x', 0): pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), ('x', 1): pd.DataFrame({'a': [4, 5, 6], 'b': [3, 2, 1]}), ('x', 2): pd.DataFrame({'a': [7, 8, 9], 'b': [0, 0, 0]})}
    meta = make_meta({'a': 'i8', 'b': 'i8'}, parent_meta=pd.DataFrame())
    ddf1 = dd.DataFrame(dsk, 'x', meta, [None, None])
    df1 = ddf1.compute()
    dsk = {('y', 0): pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60]}), ('y', 1): pd.DataFrame({'a': [40, 50, 60], 'b': [30, 20, 10]}), ('y', 2): pd.DataFrame({'a': [70, 80, 90], 'b': [0, 0, 0]})}
    ddf2 = dd.DataFrame(dsk, 'y', meta, [None, None])
    df2 = ddf2.compute()
    dsk = {('y', 0): pd.DataFrame({'b': [10, 20, 30], 'c': [40, 50, 60]}), ('y', 1): pd.DataFrame({'b': [40, 50, 60], 'c': [30, 20, 10]})}
    meta = make_meta({'b': 'i8', 'c': 'i8'}, parent_meta=pd.DataFrame())
    ddf3 = dd.DataFrame(dsk, 'y', meta, [None, None])
    df3 = ddf3.compute()
    check_append_with_warning(ddf1, ddf2, df1, df2)
    check_append_with_warning(ddf2, ddf1, df2, df1)
    check_append_with_warning(ddf1, ddf3, df1, df3)
    check_append_with_warning(ddf3, ddf1, df3, df1)
    check_append_with_warning(ddf1, df2, df1, df2)
    check_append_with_warning(ddf2, df1, df2, df1)
    check_append_with_warning(ddf1, df3, df1, df3)
    check_append_with_warning(ddf3, df1, df3, df1)