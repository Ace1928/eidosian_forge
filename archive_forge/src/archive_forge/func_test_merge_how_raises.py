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
def test_merge_how_raises():
    left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 'B': ['B0', 'B1', 'B2', 'B3']})
    right = pd.DataFrame({'A': ['C0', 'C1', 'C2', 'C3'], 'B': ['D0', 'D1', 'D2', 'D3']})
    with pytest.raises(ValueError, match="dask.dataframe.merge does not support how='cross'"):
        dd.merge(left, right, how='cross')