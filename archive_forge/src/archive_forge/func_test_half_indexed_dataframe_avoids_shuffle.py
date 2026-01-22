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
def test_half_indexed_dataframe_avoids_shuffle():
    a = pd.DataFrame({'x': np.random.randint(100, size=1000)})
    b = pd.DataFrame({'y': np.random.randint(100, size=100)}, index=np.random.randint(100, size=100))
    aa = dd.from_pandas(a, npartitions=100)
    bb = dd.from_pandas(b, npartitions=2)
    c = pd.merge(a, b, left_index=True, right_on='y')
    cc = dd.merge(aa, bb, left_index=True, right_on='y', shuffle_method='tasks')
    list_eq(c, cc)
    assert len(cc.dask) < 500