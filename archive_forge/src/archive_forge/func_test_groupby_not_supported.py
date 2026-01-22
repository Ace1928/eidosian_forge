from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
def test_groupby_not_supported():
    ddf = dd.from_pandas(pd.DataFrame({'A': [1, 1, 2, 2], 'B': [1, 2, 3, 4]}), npartitions=2)
    with pytest.raises(TypeError):
        ddf.groupby('A', axis=1)
    with pytest.raises(TypeError):
        ddf.groupby('A', level=1)
    with pytest.raises(TypeError):
        ddf.groupby('A', as_index=False)
    with pytest.raises(TypeError):
        ddf.groupby('A', squeeze=True)