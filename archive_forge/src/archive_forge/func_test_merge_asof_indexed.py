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
def test_merge_asof_indexed():
    A = pd.DataFrame({'left_val': list('abcd' * 3)}, index=[1, 3, 7, 9, 10, 13, 14, 17, 20, 24, 25, 28])
    a = dd.from_pandas(A, npartitions=4)
    B = pd.DataFrame({'right_val': list('xyz' * 4)}, index=[1, 2, 3, 6, 7, 10, 12, 14, 16, 19, 23, 26])
    b = dd.from_pandas(B, npartitions=3)
    C = pd.merge_asof(A, B, left_index=True, right_index=True)
    c = dd.merge_asof(a, b, left_index=True, right_index=True)
    assert_eq(c, C)