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
def check_and_return(ddfs, dfs, join):
    sol = concat(dfs, join=join)
    res = dd.concat(ddfs, join=join, interleave_partitions=divisions)
    assert_eq(res, sol)
    if known:
        parts = compute_as_if_collection(dd.DataFrame, res.dask, res.__dask_keys__())
        for p in [i.iloc[:0] for i in parts]:
            check_meta(res._meta, p)
    assert not cat_index or has_known_categories(res.index) == known
    return res