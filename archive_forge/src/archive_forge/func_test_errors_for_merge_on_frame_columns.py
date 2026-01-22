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
def test_errors_for_merge_on_frame_columns():
    a = pd.DataFrame({'x': [1, 2, 3, 4, 5]}, index=[1, 2, 3, 4, 5])
    b = pd.DataFrame({'y': [1, 2, 3, 4, 5]}, index=[5, 4, 3, 2, 1])
    aa = dd.from_pandas(a, npartitions=3, sort=False)
    bb = dd.from_pandas(b, npartitions=2)
    with pytest.raises(NotImplementedError):
        dd.merge(aa, bb, left_on='x', right_on=bb.y)
    with pytest.raises(NotImplementedError):
        dd.merge(aa, bb, left_on=aa.x, right_on=bb.y)