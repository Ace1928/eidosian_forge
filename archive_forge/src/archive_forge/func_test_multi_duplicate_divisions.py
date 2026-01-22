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
def test_multi_duplicate_divisions():
    df1 = pd.DataFrame({'x': [0, 0, 0, 0]})
    df2 = pd.DataFrame({'x': [0]})
    ddf1 = dd.from_pandas(df1, npartitions=2).set_index('x')
    ddf2 = dd.from_pandas(df2, npartitions=1).set_index('x')
    assert ddf1.npartitions <= 2
    assert len(ddf1) == len(df1)
    r1 = ddf1.merge(ddf2, how='left', left_index=True, right_index=True)
    sf1 = df1.set_index('x')
    sf2 = df2.set_index('x')
    r2 = sf1.merge(sf2, how='left', left_index=True, right_index=True)
    assert_eq(r1, r2)