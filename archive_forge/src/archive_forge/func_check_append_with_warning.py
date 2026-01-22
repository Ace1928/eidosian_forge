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
def check_append_with_warning(dask_obj, dask_append, pandas_obj, pandas_append):
    if PANDAS_GE_140:
        with pytest.warns(FutureWarning, match='append method is deprecated'):
            expected = pandas_obj.append(pandas_append)
            result = dask_obj.append(dask_append)
            assert_eq(result, expected)
    else:
        expected = pandas_obj.append(pandas_append)
        result = dask_obj.append(dask_append)
        assert_eq(result, expected)
    return result