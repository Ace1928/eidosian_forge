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
def test_merge_maintains_columns():
    lhs = pd.DataFrame({'A': [1, 2, 3], 'B': list('abc'), 'C': 'foo', 'D': 1.0}, columns=list('DCBA'))
    rhs = pd.DataFrame({'G': [4, 5], 'H': 6.0, 'I': 'bar', 'B': list('ab')}, columns=list('GHIB'))
    ddf = dd.from_pandas(lhs, npartitions=1)
    merged = dd.merge(ddf, rhs, on='B').compute()
    assert tuple(merged.columns) == ('D', 'C', 'B', 'A', 'G', 'H', 'I')